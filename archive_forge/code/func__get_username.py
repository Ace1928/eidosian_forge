import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _get_username(self) -> Optional[str]:
    if self._settings and self._settings._offline:
        return None
    if self._server is None:
        self._load_viewer()
    assert self._server is not None
    username = self._server._viewer.get('username')
    return username