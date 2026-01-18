import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
class LayoutLMv2VisualBackbone(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.cfg = config.get_detectron2_config()
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        self.backbone = model.backbone
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        self.register_buffer('pixel_mean', torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1), persistent=False)
        self.out_feature_key = 'p2'
        if torch.are_deterministic_algorithms_enabled():
            logger.warning('using `AvgPool2d` instead of `AdaptiveAvgPool2d`')
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            self.pool = nn.AvgPool2d((math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]), math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1])))
        else:
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    def forward(self, images):
        images_input = ((images if torch.is_tensor(images) else images.tensor) - self.pixel_mean) / self.pixel_std
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features

    def synchronize_batch_norm(self):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized() and (torch.distributed.get_rank() > -1)):
            raise RuntimeError('Make sure torch.distributed is set up properly.')
        self_rank = torch.distributed.get_rank()
        node_size = torch.cuda.device_count()
        world_size = torch.distributed.get_world_size()
        if not world_size % node_size == 0:
            raise RuntimeError('Make sure the number of processes can be divided by the number of nodes')
        node_global_ranks = [list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)]
        sync_bn_groups = [torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)]
        node_rank = self_rank // node_size
        self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])