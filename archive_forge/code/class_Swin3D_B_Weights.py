from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..swin_transformer import PatchMerging, SwinTransformerBlock
class Swin3D_B_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(url='https://download.pytorch.org/models/swin3d_b_1k-24f7c7c6.pth', transforms=partial(VideoClassification, crop_size=(224, 224), resize_size=(256,), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), meta={**_COMMON_META, 'recipe': 'https://github.com/SwinTransformer/Video-Swin-Transformer#kinetics-400', '_docs': 'The weights were ported from the paper. The accuracies are estimated on video-level with parameters `frame_rate=15`, `clips_per_video=12`, and `clip_len=32`', 'num_params': 88048984, '_metrics': {'Kinetics-400': {'acc@1': 79.427, 'acc@5': 94.386}}, '_ops': 140.667, '_file_size': 364.134})
    KINETICS400_IMAGENET22K_V1 = Weights(url='https://download.pytorch.org/models/swin3d_b_22k-7c6ae6fa.pth', transforms=partial(VideoClassification, crop_size=(224, 224), resize_size=(256,), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), meta={**_COMMON_META, 'recipe': 'https://github.com/SwinTransformer/Video-Swin-Transformer#kinetics-400', '_docs': 'The weights were ported from the paper. The accuracies are estimated on video-level with parameters `frame_rate=15`, `clips_per_video=12`, and `clip_len=32`', 'num_params': 88048984, '_metrics': {'Kinetics-400': {'acc@1': 81.643, 'acc@5': 95.574}}, '_ops': 140.667, '_file_size': 364.134})
    DEFAULT = KINETICS400_V1