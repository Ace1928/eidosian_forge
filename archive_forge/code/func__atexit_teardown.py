import atexit
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import psutil
import wandb
from wandb import env, trigger
from wandb.errors import Error
from wandb.sdk.lib.exit_hooks import ExitHooks
from wandb.sdk.lib.import_hooks import unregister_all_post_import_hooks
def _atexit_teardown(self) -> None:
    trigger.call('on_finished')
    exit_code = self._hooks.exit_code if self._hooks else 0
    self._teardown(exit_code)