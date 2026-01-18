import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
class StateShared:
    _context: StateContext

    def __init__(self) -> None:
        self._context = StateContext()

    def _update_written_offset(self, record: 'Record') -> None:
        end_offset = record.control.end_offset
        if end_offset:
            self._context.last_written_offset = end_offset

    def _update_forwarded_offset(self) -> None:
        self._context.last_forwarded_offset = self._context.last_written_offset

    def _process(self, record: 'Record') -> None:
        request_type = _get_request_type(record)
        if not request_type:
            return
        process_str = f'_process_{request_type}'
        process_handler: Optional[Callable[[pb.Record], None]] = getattr(self, process_str, None)
        if not process_handler:
            return
        process_handler(record)

    def _process_status_report(self, record: 'Record') -> None:
        sent_offset = record.request.status_report.sent_offset
        self._context.last_sent_offset = sent_offset

    def on_exit(self, record: 'Record') -> StateContext:
        return self._context

    def on_enter(self, record: 'Record', context: StateContext) -> None:
        self._context = context

    @property
    def _behind_bytes(self) -> int:
        return self._context.last_forwarded_offset - self._context.last_sent_offset