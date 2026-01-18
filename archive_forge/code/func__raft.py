from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
def _raft(*, weights=None, progress=False, feature_encoder_layers, feature_encoder_block, feature_encoder_norm_layer, context_encoder_layers, context_encoder_block, context_encoder_norm_layer, corr_block_num_levels, corr_block_radius, motion_encoder_corr_layers, motion_encoder_flow_layers, motion_encoder_out_channels, recurrent_block_hidden_state_size, recurrent_block_kernel_size, recurrent_block_padding, flow_head_hidden_size, use_mask_predictor, **kwargs):
    feature_encoder = kwargs.pop('feature_encoder', None) or FeatureEncoder(block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer)
    context_encoder = kwargs.pop('context_encoder', None) or FeatureEncoder(block=context_encoder_block, layers=context_encoder_layers, norm_layer=context_encoder_norm_layer)
    corr_block = kwargs.pop('corr_block', None) or CorrBlock(num_levels=corr_block_num_levels, radius=corr_block_radius)
    update_block = kwargs.pop('update_block', None)
    if update_block is None:
        motion_encoder = MotionEncoder(in_channels_corr=corr_block.out_channels, corr_layers=motion_encoder_corr_layers, flow_layers=motion_encoder_flow_layers, out_channels=motion_encoder_out_channels)
        out_channels_context = context_encoder_layers[-1] - recurrent_block_hidden_state_size
        recurrent_block = RecurrentBlock(input_size=motion_encoder.out_channels + out_channels_context, hidden_size=recurrent_block_hidden_state_size, kernel_size=recurrent_block_kernel_size, padding=recurrent_block_padding)
        flow_head = FlowHead(in_channels=recurrent_block_hidden_state_size, hidden_size=flow_head_hidden_size)
        update_block = UpdateBlock(motion_encoder=motion_encoder, recurrent_block=recurrent_block, flow_head=flow_head)
    mask_predictor = kwargs.pop('mask_predictor', None)
    if mask_predictor is None and use_mask_predictor:
        mask_predictor = MaskPredictor(in_channels=recurrent_block_hidden_state_size, hidden_size=256, multiplier=0.25)
    model = RAFT(feature_encoder=feature_encoder, context_encoder=context_encoder, corr_block=corr_block, update_block=update_block, mask_predictor=mask_predictor, **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model