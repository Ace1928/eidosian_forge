import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _teardown(self, exit_code: Optional[int]=None) -> None:
    exit_code = exit_code or 0
    self._teardown_manager(exit_code=exit_code)