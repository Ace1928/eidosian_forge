import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _setup_manager(self) -> None:
    if self._settings._disable_service:
        return
    self._manager = wandb_manager._Manager(settings=self._settings)