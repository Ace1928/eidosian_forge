import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
class _WandbSetup:
    """Wandb singleton class.

    Note: This is a process local singleton.
    (Forked processes will get a new copy of the object)
    """
    _instance: Optional['_WandbSetup__WandbSetup'] = None

    def __init__(self, settings: Optional[Settings]=None) -> None:
        pid = os.getpid()
        if _WandbSetup._instance and _WandbSetup._instance._pid == pid:
            _WandbSetup._instance._update(settings=settings)
            return
        _WandbSetup._instance = _WandbSetup__WandbSetup(settings=settings, pid=pid)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._instance, name)