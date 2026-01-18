import importlib
import logging
import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRScheduler, LRSchedulerConfig
def _determine_lr_attr_name(model: 'pl.LightningModule', attr_name: str='') -> str:
    if attr_name:
        if not lightning_hasattr(model, attr_name):
            raise AttributeError(f'The attribute name for the learning rate was set to {attr_name}, but could not find this as a field in `model` or `model.hparams`.')
        return attr_name
    attr_options = ('lr', 'learning_rate')
    for attr in attr_options:
        if lightning_hasattr(model, attr):
            return attr
    raise AttributeError(f'When using the learning rate finder, either `model` or `model.hparams` should have one of these fields: {attr_options}. If your model has a different name for the learning rate, set it with `.lr_find(attr_name=...)`.')