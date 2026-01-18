import atexit
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import psutil
import wandb
from wandb import env, trigger
from wandb.errors import Error
from wandb.sdk.lib.exit_hooks import ExitHooks
from wandb.sdk.lib.import_hooks import unregister_all_post_import_hooks
def _get_service_interface(self) -> 'ServiceInterface':
    assert self._service
    svc_iface = self._service.service_interface
    assert svc_iface
    return svc_iface