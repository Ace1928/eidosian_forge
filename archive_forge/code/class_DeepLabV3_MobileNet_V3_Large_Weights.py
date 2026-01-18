from functools import partial
from typing import Any, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
from .fcn import FCNHead
class DeepLabV3_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(url='https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth', transforms=partial(SemanticSegmentation, resize_size=520), meta={**_COMMON_META, 'num_params': 11029328, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_mobilenet_v3_large', '_metrics': {'COCO-val2017-VOC-labels': {'miou': 60.3, 'pixel_acc': 91.2}}, '_ops': 10.452, '_file_size': 42.301})
    DEFAULT = COCO_WITH_VOC_LABELS_V1