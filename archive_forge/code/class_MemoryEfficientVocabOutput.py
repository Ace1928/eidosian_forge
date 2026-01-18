from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
class MemoryEfficientVocabOutput(nn.Module):
    """Fused fc + softmax + nll_loss in a tiled fashion.

        MEVO uses much less memory but is quite a bit slower.

        MEVO also implements the LMCL (Large Margin Cosine Loss) function introduced by
        highly cited
        `CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`_.

        .. _`CosFace: Large Margin Cosine Loss for Deep Face Recognition [Wang et al.]`: https://arxiv.org/abs/1801.09414

        LMCL can be turned on using the ``margin`` and ``scale`` parameters below. These
        hyperparameters most likely require tuning, depending on the number of classes etc.

        MEVO LMCL can be suitable for face recognition and image retrieval tasks, esp. when
        the number prediction target classes is large. MEVO is slower but can use much
        less GPU memory in that case, which enables training with larger batches. We
        hope this is helpful but we strongly recommend users (AI researchers
        and engineers) to carefully consider their applications of this technology. This
        types of technology should not be used by small group of people exclusively to
        potentially harm the general public.

    Args:
        proj_weight (nn.Parameter):
            Sharing this weight with an embedding layer.
        tile_factor (int):
            Number of splits to use on the input sequence and vocab dimensions.
            Default: 16
        reduction (str):
            Reduction OP (sum or mean).
            Default: sum
        margin (float):
            Hyperparameter of the separation margin between classes. See the
            appendix of the CosFace paper for a formula on how to compute its
            value properly. The default value is unlikely to be suitable in all
            cases.
            Default: 0.35
        scale (Optional[float]):
            Hyperparameter of the feature-vector-scaling for LMCL. When not
            supplied, LMCL is turned off. See the appendix of the CosFace paper for
            a formula on how to compute its value properly.
            Default: None
    """

    def __init__(self, proj_weight: nn.Parameter, tile_factor: int=16, reduction: str='sum', margin: float=0.35, scale: Optional[float]=None):
        super().__init__()
        self.proj_weight = proj_weight
        self.tf_in, self.tf_w = (tile_factor, tile_factor)
        self.fp_max = True
        self.fp_sum = True
        self.fp_target = True
        self.log_softmax = True
        self.reduction = reduction
        assert self.reduction in ['sum', 'mean']
        self.margin = margin
        self.scale = scale
        self.trigger = BackwardTrigger(self.proj_weight)
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            print(f'DEBUG cfg tf_in={self.tf_in} tf_w={self.tf_w} fp_max={self.fp_max} fp_sum={self.fp_sum} fp_target={self.fp_target} log_softmax={self.log_softmax} reduction={self.reduction} margin={self.margin} scale={self.scale}')

    def get_target_nlprob(self, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, debase_max: torch.Tensor, exp_sums: torch.Tensor) -> torch.Tensor:
        """Get target's negative log probability."""
        target_score = TargetScoreFunction.apply(i, w, target, self)
        prob = (target_score - debase_max).exp() / exp_sums
        if self.log_softmax:
            prob = prob.log()
        return -prob.sum()

    def eval_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Eval time forward that doesn't fuse the softmax and NLL Loss kernels."""
        return torch.matmul(input, self.proj_weight.T)

    def forward(self, input: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.training and target is None:
            return self.eval_forward(input)
        if DEBUG and dist.is_initialized() and (dist.get_rank() == 0):
            cur_mem = round(torch.cuda.memory_allocated() / 1024 / 1024)
            mem = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
            print('DEBUG cur, peak', cur_mem, mem)
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        if torch.is_grad_enabled():
            assert input.requires_grad
        input, target = _reshape_inputs(input, target)
        tokens, d_model = input.shape
        t2, = target.shape
        vocab, d2 = self.proj_weight.shape
        assert d_model == d2, f'incorrect shape {d_model} vs {d2}'
        assert tokens == t2, f'incorrect shape {tokens} vs {t2}'
        split_dim = 0
        input_split_size = _next_power_of_2_or_max(tokens // self.tf_in, tokens)
        weight_split_size = _next_power_of_2_or_max(vocab // self.tf_w, vocab)
        inputs = torch.split(input, input_split_size, split_dim)
        weight = self.trigger()
        weights = torch.split(weight, weight_split_size, split_dim)
        targets = tuple([torch.Tensor()] * len(inputs))
        if self.scale is not None:
            targets = torch.split(target, input_split_size, split_dim)
        maxs = []
        for i, tgt in zip(inputs, targets):
            m = None
            for w_idx, w in enumerate(weights):
                _m = GetMaxFunction.apply(i, w, tgt, self, w_idx, weight_split_size, split_dim)
                if m is None:
                    m = _m
                else:
                    m = torch.max(m, _m)
            assert m is not None
            maxs.append(m)
        maxs_tensor = torch.cat(maxs)
        assert maxs_tensor.shape == (tokens,)
        sums = []
        for i, tgt, debase_max in zip(inputs, targets, maxs):
            s = None
            for w_idx, w in enumerate(weights):
                _s = GetSumFunction.apply(i, w, tgt, debase_max, self, w_idx, weight_split_size, split_dim)
                if s is None:
                    s = _s
                else:
                    s += _s
            assert s is not None
            sums.append(s)
        sums_tensor = torch.cat(sums)
        assert sums_tensor.shape == (tokens,)
        result = self.get_target_nlprob(input, self.proj_weight, target, maxs_tensor, sums_tensor)
        if self.reduction == 'mean':
            result /= tokens
        return result