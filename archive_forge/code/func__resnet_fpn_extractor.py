import warnings
from typing import Callable, Dict, List, Optional, Union
from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from .. import mobilenet, resnet
from .._api import _get_enum_from_fn, WeightsEnum
from .._utils import handle_legacy_interface, IntermediateLayerGetter
def _resnet_fpn_extractor(backbone: resnet.ResNet, trainable_layers: int, returned_layers: Optional[List[int]]=None, extra_blocks: Optional[ExtraFPNBlock]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) -> BackboneWithFPN:
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f'Trainable layers should be in the range [0,5], got {trainable_layers}')
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f'Each returned layer should be in the range [1,4]. Got {returned_layers}')
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer)