from functools import partial
from typing import Any, Optional
from torch import nn
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
class FCN_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(url='https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth', transforms=partial(SemanticSegmentation, resize_size=520), meta={**_COMMON_META, 'num_params': 35322218, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet50', '_metrics': {'COCO-val2017-VOC-labels': {'miou': 60.5, 'pixel_acc': 91.4}}, '_ops': 152.717, '_file_size': 135.009})
    DEFAULT = COCO_WITH_VOC_LABELS_V1