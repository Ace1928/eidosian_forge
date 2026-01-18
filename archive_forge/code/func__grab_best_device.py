import argparse
import os
from pathlib import Path
import torch
from accelerate.utils.modeling import find_tied_parameters
from seamless_communication.models.inference.translator import Translator
from transformers import (
from transformers.utils import logging
def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    return torch.device(device)