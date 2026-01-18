from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image, load_image
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import AddedToken, BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, logging
def _extract_images_from_prompts(self, prompts):
    prompt_images = []
    for prompt in prompts:
        images = []
        for elem in prompt:
            if is_valid_image(elem):
                images.append(elem)
            elif is_url(elem):
                images.append(load_image(elem))
        prompt_images.append(images)
    return prompt_images