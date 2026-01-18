from typing import TYPE_CHECKING, List, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_10
def _clip_score_update(images: Union[Tensor, List[Tensor]], text: Union[str, List[str]], model: _CLIPModel, processor: _CLIPProcessor) -> Tuple[Tensor, int]:
    if not isinstance(images, list):
        if images.ndim == 3:
            images = [images]
    else:
        images = list(images)
    if not all((i.ndim == 3 for i in images)):
        raise ValueError('Expected all images to be 3d but found image that has either more or less')
    if not isinstance(text, list):
        text = [text]
    if len(text) != len(images):
        raise ValueError(f'Expected the number of images and text examples to be the same but got {len(images)} and {len(text)}')
    device = images[0].device
    processed_input = processor(text=text, images=[i.cpu() for i in images], return_tensors='pt', padding=True)
    img_features = model.get_image_features(processed_input['pixel_values'].to(device))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
    max_position_embeddings = model.config.text_config.max_position_embeddings
    if processed_input['attention_mask'].shape[-1] > max_position_embeddings:
        rank_zero_warn(f'Encountered caption longer than max_position_embeddings={max_position_embeddings!r}. Will truncate captions to this length.If longer captions are needed, initialize argument `model_name_or_path` with a model that supportslonger sequences', UserWarning)
        processed_input['attention_mask'] = processed_input['attention_mask'][..., :max_position_embeddings]
        processed_input['input_ids'] = processed_input['input_ids'][..., :max_position_embeddings]
    txt_features = model.get_text_features(processed_input['input_ids'].to(device), processed_input['attention_mask'].to(device))
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)
    score = 100 * (img_features * txt_features).sum(axis=-1)
    return (score, len(text))