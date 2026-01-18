from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _PIQ_GREATER_EQUAL_0_8, _TRANSFORMERS_GREATER_EQUAL_4_10
def _download_clip_for_iqa_metric() -> None:
    _CLIPModel.from_pretrained('openai/clip-vit-base-patch16', resume_download=True)
    _CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16', resume_download=True)