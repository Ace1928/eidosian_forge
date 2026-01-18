from typing import TYPE_CHECKING, List, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_10
def _get_clip_model_and_processor(model_name_or_path: Literal['openai/clip-vit-base-patch16', 'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-large-patch14']='openai/clip-vit-large-patch14') -> Tuple[_CLIPModel, _CLIPProcessor]:
    if _TRANSFORMERS_GREATER_EQUAL_4_10:
        from transformers import CLIPModel as _CLIPModel
        from transformers import CLIPProcessor as _CLIPProcessor
        model = _CLIPModel.from_pretrained(model_name_or_path)
        processor = _CLIPProcessor.from_pretrained(model_name_or_path)
        return (model, processor)
    raise ModuleNotFoundError('`clip_score` metric requires `transformers` package be installed. Either install with `pip install transformers>=4.10.0` or `pip install torchmetrics[multimodal]`.')