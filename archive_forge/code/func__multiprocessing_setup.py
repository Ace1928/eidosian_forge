import importlib.machinery
import logging
import multiprocessing
import os
import queue
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union, cast
import wandb
from wandb.sdk.interface.interface import InterfaceBase
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal.internal import wandb_internal
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib.mailbox import Mailbox
from wandb.sdk.wandb_manager import _Manager
from wandb.sdk.wandb_settings import Settings
def _multiprocessing_setup(self) -> None:
    assert self._settings
    if self._settings.start_method == 'thread':
        return
    start_method = self._settings.start_method or 'spawn'
    if not hasattr(multiprocessing, 'get_context'):
        return
    all_methods = multiprocessing.get_all_start_methods()
    logger.info('multiprocessing start_methods={}, using: {}'.format(','.join(all_methods), start_method))
    ctx = multiprocessing.get_context(start_method)
    self._multiprocessing = ctx