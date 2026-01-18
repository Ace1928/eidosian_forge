import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_vision_available, logging
class FlavaMaskingGenerator:

    def __init__(self, input_size: Union[int, Tuple[int, int]]=14, total_mask_patches: int=75, mask_group_max_patches: Optional[int]=None, mask_group_min_patches: int=16, mask_group_min_aspect_ratio: Optional[float]=0.3, mask_group_max_aspect_ratio: float=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.total_mask_patches = total_mask_patches
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = total_mask_patches if mask_group_max_patches is None else mask_group_max_patches
        mask_group_max_aspect_ratio = mask_group_max_aspect_ratio or 1 / mask_group_min_aspect_ratio
        self.log_aspect_ratio = (math.log(mask_group_min_aspect_ratio), math.log(mask_group_max_aspect_ratio))

    def __repr__(self):
        repr_str = 'MaskingGenerator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)' % (self.height, self.width, self.mask_group_min_patches, self.mask_group_max_patches, self.total_mask_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return (self.height, self.width)

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _attempt in range(10):
            target_area = random.uniform(self.mask_group_min_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)
                num_masked = mask[top:top + height, left:left + width].sum()
                if 0 < height * width - num_masked <= max_mask_patches:
                    for i in range(top, top + height):
                        for j in range(left, left + width):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        mask_count = 0
        while mask_count < self.total_mask_patches:
            max_mask_patches = self.total_mask_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.mask_group_max_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask