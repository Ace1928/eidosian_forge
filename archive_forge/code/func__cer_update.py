from typing import List, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _edit_distance
def _cer_update(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
    """Update the cer score with the current set of references and predictions.

    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings

    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of character overall references

    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = tensor(0, dtype=torch.float)
    total = tensor(0, dtype=torch.float)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred
        tgt_tokens = tgt
        errors += _edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)
    return (errors, total)