import enum
import os
import sys
from typing import Dict, Optional, Tuple
import click
import wandb
from wandb.errors import AuthenticationError, UsageError
from wandb.old.settings import Settings as OldSettings
from ..apis import InternalApi
from .internal.internal_api import Api
from .lib import apikey
from .wandb_settings import Settings, Source
def _handle_host_wandb_setting(host: Optional[str], cloud: bool=False) -> None:
    """Write the host parameter to the global settings file.

    This takes the parameter from wandb.login or wandb login for use by the
    application's APIs.
    """
    _api = InternalApi()
    if host == 'https://api.wandb.ai' or (host is None and cloud):
        _api.clear_setting('base_url', globally=True, persist=True)
        if os.path.exists(OldSettings._local_path()):
            _api.clear_setting('base_url', persist=True)
    elif host:
        host = host.rstrip('/')
        _api.set_setting('base_url', host, globally=True, persist=True)