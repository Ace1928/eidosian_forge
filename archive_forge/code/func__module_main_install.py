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
def _module_main_install(self) -> None:
    main_module = sys.modules['__main__']
    main_mod_spec = getattr(main_module, '__spec__', None)
    main_mod_path = getattr(main_module, '__file__', None)
    if main_mod_spec is None:
        loader: Loader = importlib.machinery.BuiltinImporter
        main_mod_spec = importlib.machinery.ModuleSpec(name='wandb.mpmain', loader=loader)
        main_module.__spec__ = main_mod_spec
    else:
        self._save_mod_spec = main_mod_spec
    if main_mod_path is not None:
        self._save_mod_path = main_module.__file__
        fname = os.path.join(os.path.dirname(wandb.__file__), 'mpmain', '__main__.py')
        main_module.__file__ = fname