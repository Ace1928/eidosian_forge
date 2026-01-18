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
class OriginalMaskFormerConfigToImageProcessorConverter:

    def __call__(self, original_config: object) -> MaskFormerImageProcessor:
        model = original_config.MODEL
        model_input = original_config.INPUT
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST[0])
        return MaskFormerImageProcessor(image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(), image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(), size=model_input.MIN_SIZE_TEST, max_size=model_input.MAX_SIZE_TEST, num_labels=model.SEM_SEG_HEAD.NUM_CLASSES, ignore_index=dataset_catalog.ignore_label, size_divisibility=32)