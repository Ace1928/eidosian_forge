import warnings
from typing import List, Optional, Union
import numpy as np
import PIL
import torch
from diffusers import ConfigMixin
from diffusers.image_processor import VaeImageProcessor as DiffusersVaeImageProcessor
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from PIL import Image
from tqdm.auto import tqdm
class DiffusionPipelineMixin(ConfigMixin):

    @staticmethod
    def numpy_to_pil(images):
        """
        Converts a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype('uint8')
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode='L') for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, '_progress_bar_config'):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(f'`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}.')
        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError('Either `total` or `iterable` has to be defined.')