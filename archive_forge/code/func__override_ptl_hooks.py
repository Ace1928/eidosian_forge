import inspect
import logging
import os
import tempfile
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Type, Union
from pytorch_lightning import Callback, Trainer, LightningModule
from ray import train
from ray.util import log_once
from ray.util.annotations import PublicAPI, Deprecated
from ray.train import Checkpoint
def _override_ptl_hooks(callback_cls: Type['TuneCallback']) -> Type['TuneCallback']:
    """Overrides all allowed PTL Callback hooks with our custom handle logic."""

    def generate_overridden_hook(fn_name):

        def overridden_hook(self, trainer: Trainer, *args, pl_module: Optional[LightningModule]=None, **kwargs):
            if fn_name in self._on:
                self._handle(trainer=trainer, pl_module=pl_module)
        return overridden_hook
    for fn_name in _allowed_hooks:
        setattr(callback_cls, fn_name, generate_overridden_hook(fn_name))
    return callback_cls