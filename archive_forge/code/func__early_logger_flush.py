import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _early_logger_flush(self, new_logger: Logger) -> None:
    if not self._early_logger:
        return
    _set_logger(new_logger)
    self._early_logger._flush()