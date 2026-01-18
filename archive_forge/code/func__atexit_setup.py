import atexit
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import psutil
import wandb
from wandb import env, trigger
from wandb.errors import Error
from wandb.sdk.lib.exit_hooks import ExitHooks
from wandb.sdk.lib.import_hooks import unregister_all_post_import_hooks
def _atexit_setup(self) -> None:
    self._atexit_lambda = lambda: self._atexit_teardown()
    self._hooks = ExitHooks()
    self._hooks.hook()
    atexit.register(self._atexit_lambda)