from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _PIQ_GREATER_EQUAL_0_8, _TRANSFORMERS_GREATER_EQUAL_4_10
def _clip_iqa_compute(img_features: Tensor, anchors: Tensor, prompts_names: List[str], format_as_dict: bool=True) -> Union[Tensor, Dict[str, Tensor]]:
    """Final computation of CLIP IQA."""
    logits_per_image = 100 * img_features @ anchors.t()
    probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(-1)[:, :, 0]
    if len(prompts_names) == 1:
        return probs.squeeze()
    if format_as_dict:
        return {p: probs[:, i] for i, p in enumerate(prompts_names)}
    return probs