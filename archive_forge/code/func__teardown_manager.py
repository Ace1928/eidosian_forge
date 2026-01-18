import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _teardown_manager(self, exit_code: int) -> None:
    if not self._manager:
        return
    self._manager._teardown(exit_code)
    self._manager = None