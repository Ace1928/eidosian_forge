import argparse
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, cast
import pytorch_lightning as pl
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from xformers.benchmarks.LRA.code.dataset import LRADataset
from xformers.benchmarks.LRA.code.model_wrapper import ModelForSC, ModelForSCDual
from xformers.components.attention import ATTENTION_REGISTRY
class BasicProgressBar(TQDMProgressBar):

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop('v_num', None)
        return items