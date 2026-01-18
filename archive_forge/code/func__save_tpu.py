import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .integrations import (
import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
def _save_tpu(self, output_dir: Optional[str]=None):
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    logger.info(f'Saving model checkpoint to {output_dir}')
    model = self.model
    model.to('cpu')
    if xm.is_master_ordinal():
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    xm.rendezvous('saving_checkpoint')
    if not isinstance(model, PreTrainedModel):
        if isinstance(unwrap_model(model), PreTrainedModel):
            unwrap_model(model).save_pretrained(output_dir, is_main_process=self.args.should_save, state_dict=model.state_dict(), save_function=xm.save, safe_serialization=self.args.save_safetensors)
        else:
            logger.info('Trainer.model is not a `PreTrainedModel`, only saving its state dict.')
            state_dict = model.state_dict()
            xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        model.save_pretrained(output_dir, is_main_process=self.args.should_save, save_function=xm.save, safe_serialization=self.args.save_safetensors)
    if self.tokenizer is not None and self.args.should_save:
        self.tokenizer.save_pretrained(output_dir)
    model.to(self.args.device)