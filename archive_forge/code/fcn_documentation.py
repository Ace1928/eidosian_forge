from functools import partial
from typing import Any, Optional
from torch import nn
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
Fully-Convolutional Network model with a ResNet-101 backbone from the `Fully Convolutional
    Networks for Semantic Segmentation <https://arxiv.org/abs/1411.4038>`_ paper.

    .. betastatus:: segmentation module

    Args:
        weights (:class:`~torchvision.models.segmentation.FCN_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.FCN_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.fcn.FCN``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.segmentation.FCN_ResNet101_Weights
        :members:
    