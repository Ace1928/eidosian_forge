from functools import partial
from typing import Any, cast, Dict, List, Optional, Union
import torch
import torch.nn as nn
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class VGG16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/vgg16-397923af.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 138357544, '_metrics': {'ImageNet-1K': {'acc@1': 71.592, 'acc@5': 90.382}}, '_ops': 15.47, '_file_size': 527.796})
    IMAGENET1K_FEATURES = Weights(url='https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth', transforms=partial(ImageClassification, crop_size=224, mean=(0.48235, 0.45882, 0.40784), std=(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0)), meta={**_COMMON_META, 'num_params': 138357544, 'categories': None, 'recipe': 'https://github.com/amdegroot/ssd.pytorch#training-ssd', '_metrics': {'ImageNet-1K': {'acc@1': float('nan'), 'acc@5': float('nan')}}, '_ops': 15.47, '_file_size': 527.802, '_docs': "\n                These weights can't be used for classification because they are missing values in the `classifier`\n                module. Only the `features` module has valid values and can be used for feature extraction. The weights\n                were trained using the original input standardization method as described in the paper.\n            "})
    DEFAULT = IMAGENET1K_V1