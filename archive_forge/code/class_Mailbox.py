import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
class Mailbox:
    _slots: Dict[str, _MailboxSlot]
    _keepalive: bool

    def __init__(self) -> None:
        self._slots = {}
        self._keepalive = False

    def enable_keepalive(self) -> None:
        self._keepalive = True

    def wait(self, handle: MailboxHandle, *, timeout: float, on_progress: Optional[Callable[[MailboxProgress], None]]=None, cancel: bool=False) -> Optional[pb.Result]:
        return handle.wait(timeout=timeout, on_progress=on_progress, cancel=cancel)

    def _time(self) -> float:
        return time.monotonic()

    def wait_all(self, handles: List[MailboxHandle], *, timeout: float, on_progress_all: Optional[Callable[[MailboxProgressAll], None]]=None) -> bool:
        progress_all_handle: Optional[MailboxProgressAll] = None
        if on_progress_all:
            progress_all_handle = MailboxProgressAll()
        wait_all = _MailboxWaitAll()
        for handle in handles:
            wait_all._add_handle(handle)
            if progress_all_handle and handle._on_progress:
                progress_handle = MailboxProgress(_handle=handle)
                if handle._on_probe:
                    probe_handle = MailboxProbe()
                    progress_handle.add_probe_handle(probe_handle)
                progress_all_handle.add_progress_handle(progress_handle)
        start_time = self._time()
        while wait_all.active_handles_count > 0:
            if self._keepalive:
                for handle in wait_all.active_handles:
                    if not handle._interface:
                        continue
                    if handle._interface._transport_keepalive_failed():
                        wait_all._mark_handle_failed(handle)
                if not wait_all.active_handles_count:
                    if wait_all.failed_handles_count:
                        wait_all.clear_handles()
                        raise MailboxError('transport failed')
                    break
            wait_all._get_and_clear(timeout=1)
            if progress_all_handle and on_progress_all:
                for progress_handle in progress_all_handle.get_progress_handles():
                    for probe_handle in progress_handle.get_probe_handles():
                        if progress_handle._handle and progress_handle._handle._on_probe:
                            progress_handle._handle._on_probe(probe_handle)
                on_progress_all(progress_all_handle)
            now = self._time()
            if timeout >= 0 and now >= start_time + timeout:
                break
        return wait_all.active_handles_count == 0

    def deliver(self, result: pb.Result) -> None:
        mailbox = result.control.mailbox_slot
        slot = self._slots.get(mailbox)
        if not slot:
            return
        slot._deliver(result)

    def _allocate_slot(self) -> _MailboxSlot:
        address = _generate_address()
        slot = _MailboxSlot(address=address)
        self._slots[address] = slot
        return slot

    def _release_slot(self, address: str) -> None:
        self._slots.pop(address, None)

    def get_handle(self) -> MailboxHandle:
        slot = self._allocate_slot()
        handle = MailboxHandle(mailbox=self, slot=slot)
        return handle

    def _deliver_record(self, record: pb.Record, interface: 'InterfaceShared') -> MailboxHandle:
        handle = self.get_handle()
        handle._interface = interface
        handle._keepalive = self._keepalive
        record.control.mailbox_slot = handle.address
        try:
            interface._publish(record)
        except Exception:
            interface._transport_mark_failed()
            raise
        interface._transport_mark_success()
        return handle