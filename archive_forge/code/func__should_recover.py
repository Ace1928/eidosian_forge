import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
def _should_recover(self, record: 'Record') -> bool:
    return self._behind_bytes < self._threshold_recover