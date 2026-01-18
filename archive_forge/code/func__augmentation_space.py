import math
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from . import functional as F, InterpolationMode
def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
    s = {'ShearX': (torch.linspace(0.0, 0.3, num_bins), True), 'ShearY': (torch.linspace(0.0, 0.3, num_bins), True), 'TranslateX': (torch.linspace(0.0, image_size[1] / 3.0, num_bins), True), 'TranslateY': (torch.linspace(0.0, image_size[0] / 3.0, num_bins), True), 'Rotate': (torch.linspace(0.0, 30.0, num_bins), True), 'Posterize': (4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False), 'Solarize': (torch.linspace(255.0, 0.0, num_bins), False), 'AutoContrast': (torch.tensor(0.0), False), 'Equalize': (torch.tensor(0.0), False)}
    if self.all_ops:
        s.update({'Brightness': (torch.linspace(0.0, 0.9, num_bins), True), 'Color': (torch.linspace(0.0, 0.9, num_bins), True), 'Contrast': (torch.linspace(0.0, 0.9, num_bins), True), 'Sharpness': (torch.linspace(0.0, 0.9, num_bins), True)})
    return s