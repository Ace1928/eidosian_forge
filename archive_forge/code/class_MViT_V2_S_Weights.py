import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch
import torch.fx
import torch.nn as nn
from ...ops import MLP, StochasticDepth
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
class MViT_V2_S_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(url='https://download.pytorch.org/models/mvit_v2_s-ae3be167.pth', transforms=partial(VideoClassification, crop_size=(224, 224), resize_size=(256,), mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)), meta={'min_size': (224, 224), 'min_temporal_size': 16, 'categories': _KINETICS400_CATEGORIES, 'recipe': 'https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md', '_docs': 'The weights were ported from the paper. The accuracies are estimated on video-level with parameters `frame_rate=7.5`, `clips_per_video=5`, and `clip_len=16`', 'num_params': 34537744, '_metrics': {'Kinetics-400': {'acc@1': 80.757, 'acc@5': 94.665}}, '_ops': 64.224, '_file_size': 131.884})
    DEFAULT = KINETICS400_V1