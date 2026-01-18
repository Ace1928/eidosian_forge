import math
import warnings
from typing import Any, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from peft.tuners.lycoris_utils import LycorisLayer, check_adapters_to_merge
class OFTLayer(nn.Module, LycorisLayer):
    adapter_layer_names = ('oft_r',)

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)
        self.oft_r = nn.ParameterDict({})
        self.coft = {}
        self.eps = {}
        self.block_share = {}

    @property
    def _available_adapters(self) -> Set[str]:
        return {*self.oft_r}

    def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...], block_share: bool):
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(1, math.ceil(shape[0] / r), math.ceil(shape[0] / r)))
        else:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(r, math.ceil(shape[0] / r), math.ceil(shape[0] / r)))

    def reset_adapter_parameters(self, adapter_name: str):
        nn.init.zeros_(self.oft_r[adapter_name])

    def reset_adapter_parameters_random(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.oft_r[adapter_name], a=math.sqrt(5))

    def update_layer(self, adapter_name: str, r: int, module_dropout: float, init_weights: bool, coft: bool=False, eps: float=6e-05, block_share: bool=False, **kwargs) -> None:
        """Internal function to create oft adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
            coft (`bool`): Whether to use the constrained variant of OFT or not.
            eps (`float`):
                The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
            block_share (`bool`): Whether to share the OFT parameters between blocks or not.
        """
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.module_dropout[adapter_name] = module_dropout
        self.coft[adapter_name] = coft
        self.block_share[adapter_name] = block_share
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            shape = (base_layer.out_channels, base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1])
        else:
            raise TypeError(f'OFT is not implemented for base layers of type {type(base_layer).__name__}')
        self.eps[adapter_name] = eps * math.ceil(shape[0] / r) * math.ceil(shape[0] / r)
        self.create_adapter_parameters(adapter_name, r, shape, block_share)
        if init_weights:
            self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)
        weight = getattr(self.get_base_layer(), 'weight', None)
        if weight is not None:
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def unscale_layer(self, scale=None) -> None:
        pass

    def merge(self, safe_merge: bool=False, adapter_names: Optional[List[str]]=None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                orig_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = orig_weights.view([base_layer.out_channels, base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1]])
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                delta_weight = self.get_delta_weight(active_adapter)
                if orig_weights.shape[1] != delta_weight.shape[1]:
                    delta_weight = delta_weight[:orig_weights.shape[1], :orig_weights.shape[1]]
                new_weights = torch.mm(orig_weights, delta_weight)
                if isinstance(base_layer, nn.Linear):
                    new_weights = torch.transpose(new_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    new_weights = torch.transpose(new_weights, 0, 1)
                    new_weights = new_weights.view([base_layer.out_channels, base_layer.in_channels, base_layer.kernel_size[0], base_layer.kernel_size[1]])
                if safe_merge and (not torch.isfinite(new_weights).all()):
                    raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                base_layer.weight.data = new_weights
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                new_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    new_weights = torch.transpose(new_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    new_weights = new_weights.view([base_layer.out_channels, base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1]])
                    new_weights = torch.transpose(new_weights, 0, 1)
                delta_weight = self.get_delta_weight(active_adapter)
                if new_weights.shape[1] != delta_weight.shape[1]:
                    delta_weight = delta_weight[:new_weights.shape[1], :new_weights.shape[1]]
                delta_inv = torch.inverse(delta_weight)
                orig_weights = torch.mm(new_weights, delta_inv)
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights.reshape([base_layer.out_channels, base_layer.in_channels, base_layer.kernel_size[0], base_layer.kernel_size[1]])
                base_layer.weight.data = orig_weights

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        rank = self.r[adapter_name]
        coft = self.coft[adapter_name]
        eps = self.eps[adapter_name]
        opt_r = self.oft_r[adapter_name]
        if coft:
            with torch.no_grad():
                opt_r.copy_(self._project_batch(opt_r, eps=eps))
        orth_rotate = self._cayley_batch(opt_r)
        weight = self._block_diagonal(orth_rotate, rank)
        return weight

    def _cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
        b, r, c = data.shape
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)
        Q = torch.bmm(I - skew, torch.inverse(I + skew))
        return Q

    def _block_diagonal(self, oft_r: torch.Tensor, rank: int) -> torch.Tensor:
        if oft_r.shape[0] == 1:
            blocks = [oft_r[0, ...] for i in range(rank)]
        else:
            blocks = [oft_r[i, ...] for i in range(rank)]
        A = torch.block_diag(*blocks)
        return A

    def _project_batch(self, oft_r, eps=1e-05):
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0]))
        I = torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype).unsqueeze(0).expand_as(oft_r)
        diff = oft_r - I
        norm_diff = torch.norm(oft_r - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_r, I + eps * (diff / norm_diff))
        return out

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            if len(result.shape) == 4:
                result = result.permute(0, 2, 3, 1)
            base_layer = self.get_base_layer()
            base_bias = base_layer.bias
            if base_bias is not None:
                result = result - base_bias.data
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue
                module_dropout = self.module_dropout[active_adapter]
                if not self.training or (self.training and torch.rand(1) > module_dropout):
                    result = self._get_delta_activations(active_adapter, result, *args, **kwargs)
            if base_bias is not None:
                result = result + base_bias.data
            if len(result.shape) == 4:
                result = result.permute(0, 3, 1, 2)
        result = result.to(previous_dtype)
        return result