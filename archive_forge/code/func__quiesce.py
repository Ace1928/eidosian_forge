import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
def _quiesce(self, record: 'Record') -> None:
    start = self._context.last_forwarded_offset
    end = self._context.last_written_offset
    if start != end:
        self._recover_records(start, end)
    if _is_local_non_control_record(record):
        self._forward_record(record)
    self._update_forwarded_offset()