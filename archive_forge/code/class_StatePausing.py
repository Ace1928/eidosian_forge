import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
class StatePausing(StateShared):
    _forward_record: Callable[['Record'], None]
    _recover_records: Callable[[int, int], None]
    _threshold_recover: int
    _threshold_forward: int

    def __init__(self, forward_record: Callable[['Record'], None], recover_records: Callable[[int, int], None], threshold_recover: int, threshold_forward: int) -> None:
        super().__init__()
        self._forward_record = forward_record
        self._recover_records = recover_records
        self._threshold_recover = threshold_recover
        self._threshold_forward = threshold_forward

    def _should_unpause(self, record: 'Record') -> bool:
        return self._behind_bytes < self._threshold_forward

    def _unpause(self, record: 'Record') -> None:
        self._quiesce(record)

    def _should_recover(self, record: 'Record') -> bool:
        return self._behind_bytes < self._threshold_recover

    def _recover(self, record: 'Record') -> None:
        self._quiesce(record)

    def _should_quiesce(self, record: 'Record') -> bool:
        return _is_local_non_control_record(record)

    def _quiesce(self, record: 'Record') -> None:
        start = self._context.last_forwarded_offset
        end = self._context.last_written_offset
        if start != end:
            self._recover_records(start, end)
        if _is_local_non_control_record(record):
            self._forward_record(record)
        self._update_forwarded_offset()

    def on_check(self, record: 'Record') -> None:
        self._update_written_offset(record)
        self._process(record)