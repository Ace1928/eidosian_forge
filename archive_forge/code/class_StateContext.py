import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
@dataclass
class StateContext:
    last_forwarded_offset: int = 0
    last_sent_offset: int = 0
    last_written_offset: int = 0