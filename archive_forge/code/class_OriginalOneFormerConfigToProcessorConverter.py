import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn
from transformers import CLIPTokenizer, DinatConfig, SwinConfig
from transformers.models.oneformer.image_processing_oneformer import OneFormerImageProcessor
from transformers.models.oneformer.modeling_oneformer import (
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
from transformers.utils import logging
class OriginalOneFormerConfigToProcessorConverter:

    def __call__(self, original_config: object, model_repo: str) -> OneFormerProcessor:
        model = original_config.MODEL
        model_input = original_config.INPUT
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST_PANOPTIC[0])
        if 'ade20k' in model_repo:
            class_info_file = 'ade20k_panoptic.json'
        elif 'coco' in model_repo:
            class_info_file = 'coco_panoptic.json'
        elif 'cityscapes' in model_repo:
            class_info_file = 'cityscapes_panoptic.json'
        else:
            raise ValueError('Invalid Dataset!')
        image_processor = OneFormerImageProcessor(image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(), image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(), size=model_input.MIN_SIZE_TEST, max_size=model_input.MAX_SIZE_TEST, num_labels=model.SEM_SEG_HEAD.NUM_CLASSES, ignore_index=dataset_catalog.ignore_label, class_info_file=class_info_file)
        tokenizer = CLIPTokenizer.from_pretrained(model_repo)
        return OneFormerProcessor(image_processor=image_processor, tokenizer=tokenizer, task_seq_length=original_config.INPUT.TASK_SEQ_LEN, max_seq_length=original_config.INPUT.MAX_SEQ_LEN)