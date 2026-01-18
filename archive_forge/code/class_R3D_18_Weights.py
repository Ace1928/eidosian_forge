from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch.nn as nn
from torch import Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .._utils import _ModelURLs
class R3D_18_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(url='https://download.pytorch.org/models/r3d_18-b3b3357e.pth', transforms=partial(VideoClassification, crop_size=(112, 112), resize_size=(128, 171)), meta={**_COMMON_META, 'num_params': 33371472, '_metrics': {'Kinetics-400': {'acc@1': 63.2, 'acc@5': 83.479}}, '_ops': 40.697, '_file_size': 127.359})
    DEFAULT = KINETICS400_V1