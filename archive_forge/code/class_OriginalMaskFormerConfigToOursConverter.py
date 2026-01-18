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
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from PIL import Image
from torch import Tensor, nn
from transformers.models.maskformer.feature_extraction_maskformer import MaskFormerImageProcessor
from transformers.models.maskformer.modeling_maskformer import (
from transformers.utils import logging
class OriginalMaskFormerConfigToOursConverter:

    def __call__(self, original_config: object) -> MaskFormerConfig:
        model = original_config.MODEL
        mask_former = model.MASK_FORMER
        swin = model.SWIN
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST[0])
        id2label = dict(enumerate(dataset_catalog.stuff_classes))
        label2id = {label: idx for idx, label in id2label.items()}
        config: MaskFormerConfig = MaskFormerConfig(fpn_feature_size=model.SEM_SEG_HEAD.CONVS_DIM, mask_feature_size=model.SEM_SEG_HEAD.MASK_DIM, num_labels=model.SEM_SEG_HEAD.NUM_CLASSES, no_object_weight=mask_former.NO_OBJECT_WEIGHT, num_queries=mask_former.NUM_OBJECT_QUERIES, backbone_config={'pretrain_img_size': swin.PRETRAIN_IMG_SIZE, 'image_size': swin.PRETRAIN_IMG_SIZE, 'in_channels': 3, 'patch_size': swin.PATCH_SIZE, 'embed_dim': swin.EMBED_DIM, 'depths': swin.DEPTHS, 'num_heads': swin.NUM_HEADS, 'window_size': swin.WINDOW_SIZE, 'drop_path_rate': swin.DROP_PATH_RATE, 'model_type': 'swin'}, dice_weight=mask_former.DICE_WEIGHT, ce_weight=1.0, mask_weight=mask_former.MASK_WEIGHT, decoder_config={'model_type': 'detr', 'max_position_embeddings': 1024, 'encoder_layers': 6, 'encoder_ffn_dim': 2048, 'encoder_attention_heads': 8, 'decoder_layers': mask_former.DEC_LAYERS, 'decoder_ffn_dim': mask_former.DIM_FEEDFORWARD, 'decoder_attention_heads': mask_former.NHEADS, 'encoder_layerdrop': 0.0, 'decoder_layerdrop': 0.0, 'd_model': mask_former.HIDDEN_DIM, 'dropout': mask_former.DROPOUT, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'init_std': 0.02, 'init_xavier_std': 1.0, 'scale_embedding': False, 'auxiliary_loss': False, 'dilation': False}, id2label=id2label, label2id=label2id)
        return config