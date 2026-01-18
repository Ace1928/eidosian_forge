import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn
from transformers import (
from transformers.models.mask2former.modeling_mask2former import (
from transformers.utils import logging
class OriginalMask2FormerConfigToOursConverter:

    def __call__(self, original_config: object) -> Mask2FormerConfig:
        model = original_config.MODEL
        repo_id = 'huggingface/label-files'
        if model.SEM_SEG_HEAD.NUM_CLASSES == 847:
            filename = 'mask2former-ade20k-full-id2label.json'
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 150:
            filename = 'ade20k-id2label.json'
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 80:
            filename = 'coco-detection-mmdet-id2label.json'
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 171:
            filename = 'mask2former-coco-stuff-id2label.json'
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 133:
            filename = 'coco-panoptic-id2label.json'
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 19:
            filename = 'cityscapes-id2label.json'
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 8:
            filename = 'cityscapes-instance-id2label.json'
        elif model.SEM_SEG_HEAD.NUM_CLASSES == 65:
            filename = 'mapillary-vistas-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {label: idx for idx, label in id2label.items()}
        if model.SWIN.EMBED_DIM == 96:
            backbone_config = SwinConfig.from_pretrained('microsoft/swin-tiny-patch4-window7-224', out_features=['stage1', 'stage2', 'stage3', 'stage4'])
        elif model.SWIN.EMBED_DIM == 128:
            backbone_config = SwinConfig(embed_dim=128, window_size=12, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), out_features=['stage1', 'stage2', 'stage3', 'stage4'])
        elif model.SWIN.EMBED_DIM == 192:
            backbone_config = SwinConfig.from_pretrained('microsoft/swin-large-patch4-window12-384', out_features=['stage1', 'stage2', 'stage3', 'stage4'])
        else:
            raise ValueError(f'embed dim {model.SWIN.EMBED_DIM} not supported for Swin!')
        backbone_config.drop_path_rate = model.SWIN.DROP_PATH_RATE
        backbone_config.attention_probs_dropout_prob = model.SWIN.ATTN_DROP_RATE
        backbone_config.depths = model.SWIN.DEPTHS
        config: Mask2FormerConfig = Mask2FormerConfig(ignore_value=model.SEM_SEG_HEAD.IGNORE_VALUE, num_labels=model.SEM_SEG_HEAD.NUM_CLASSES, num_queries=model.MASK_FORMER.NUM_OBJECT_QUERIES, no_object_weight=model.MASK_FORMER.NO_OBJECT_WEIGHT, class_weight=model.MASK_FORMER.CLASS_WEIGHT, mask_weight=model.MASK_FORMER.MASK_WEIGHT, dice_weight=model.MASK_FORMER.DICE_WEIGHT, train_num_points=model.MASK_FORMER.TRAIN_NUM_POINTS, oversample_ratio=model.MASK_FORMER.OVERSAMPLE_RATIO, importance_sample_ratio=model.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO, init_std=0.02, init_xavier_std=1.0, use_auxiliary_loss=model.MASK_FORMER.DEEP_SUPERVISION, feature_strides=[4, 8, 16, 32], backbone_config=backbone_config, id2label=id2label, label2id=label2id, feature_size=model.SEM_SEG_HEAD.CONVS_DIM, mask_feature_size=model.SEM_SEG_HEAD.MASK_DIM, hidden_dim=model.MASK_FORMER.HIDDEN_DIM, encoder_layers=model.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS, encoder_feedforward_dim=1024, decoder_layers=model.MASK_FORMER.DEC_LAYERS, num_attention_heads=model.MASK_FORMER.NHEADS, dropout=model.MASK_FORMER.DROPOUT, dim_feedforward=model.MASK_FORMER.DIM_FEEDFORWARD, pre_norm=model.MASK_FORMER.PRE_NORM, enforce_input_proj=model.MASK_FORMER.ENFORCE_INPUT_PROJ, common_stride=model.SEM_SEG_HEAD.COMMON_STRIDE)
        return config