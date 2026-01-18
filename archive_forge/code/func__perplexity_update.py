from typing import Optional, Tuple
import torch
from torch import Tensor
def _perplexity_update(preds: Tensor, target: Tensor, ignore_index: Optional[int]=None) -> Tuple[Tensor, Tensor]:
    """Compute intermediate statistics for Perplexity.

    Args:
        preds:
            Logits or a unnormalized score assigned to each token in a sequence with shape [batch_size, seq_len,
            vocab_size]. Scores will be normalized internally using softmax.
        target:
            Ground truth values with a shape [batch_size, seq_len].
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.

    Returns:
        Log probabilities, summed over all samples
        Number of samples

    """
    _check_shape_and_type_consistency(preds, target)
    probs = torch.nn.functional.softmax(preds.reshape(-1, preds.shape[-1]), dim=1)
    target = target.reshape(-1)
    if ignore_index is not None:
        mask = target.ne(ignore_index)
        target = target.where(target != ignore_index, torch.tensor(0, device=target.device))
    else:
        mask = torch.ones_like(target, dtype=torch.bool)
    probs = probs[torch.arange(target.numel()), target][mask]
    total_log_probs = -probs.log().sum()
    count = mask.sum()
    return (total_log_probs, count)