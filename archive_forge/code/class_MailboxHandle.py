import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
class MailboxHandle:
    _mailbox: 'Mailbox'
    _slot: _MailboxSlot
    _on_probe: Optional[Callable[[MailboxProbe], None]]
    _on_progress: Optional[Callable[[MailboxProgress], None]]
    _interface: Optional['InterfaceShared']
    _keepalive: bool
    _failed: bool

    def __init__(self, mailbox: 'Mailbox', slot: _MailboxSlot) -> None:
        self._mailbox = mailbox
        self._slot = slot
        self._on_probe = None
        self._on_progress = None
        self._interface = None
        self._keepalive = False
        self._failed = False

    def add_probe(self, on_probe: Callable[[MailboxProbe], None]) -> None:
        self._on_probe = on_probe

    def add_progress(self, on_progress: Callable[[MailboxProgress], None]) -> None:
        self._on_progress = on_progress

    def _time(self) -> float:
        return time.monotonic()

    def wait(self, *, timeout: float, on_probe: Optional[Callable[[MailboxProbe], None]]=None, on_progress: Optional[Callable[[MailboxProgress], None]]=None, release: bool=True, cancel: bool=False) -> Optional[pb.Result]:
        probe_handle: Optional[MailboxProbe] = None
        progress_handle: Optional[MailboxProgress] = None
        found: Optional[pb.Result] = None
        start_time = self._time()
        percent_done = 0.0
        progress_sent = False
        wait_timeout = 1.0
        if timeout >= 0:
            wait_timeout = min(timeout, wait_timeout)
        on_progress = on_progress or self._on_progress
        if on_progress:
            progress_handle = MailboxProgress(_handle=self)
        on_probe = on_probe or self._on_probe
        if on_probe:
            probe_handle = MailboxProbe()
            if progress_handle:
                progress_handle.add_probe_handle(probe_handle)
        while True:
            if self._keepalive and self._interface:
                if self._interface._transport_keepalive_failed():
                    raise MailboxError('transport failed')
            found, abandoned = self._slot._get_and_clear(timeout=wait_timeout)
            if found:
                if on_progress and progress_handle and progress_sent:
                    progress_handle.set_percent_done(100)
                    on_progress(progress_handle)
                break
            if abandoned:
                break
            now = self._time()
            if timeout >= 0:
                if now >= start_time + timeout:
                    break
            if on_probe and probe_handle:
                on_probe(probe_handle)
            if on_progress and progress_handle:
                if timeout > 0:
                    percent_done = min((now - start_time) / timeout, 1.0)
                progress_handle.set_percent_done(percent_done)
                on_progress(progress_handle)
                if progress_handle._is_stopped:
                    break
                progress_sent = True
        if not found and cancel:
            self._cancel()
        if release:
            self._release()
        return found

    def _cancel(self) -> None:
        mailbox_slot = self.address
        if self._interface:
            self._interface.publish_cancel(mailbox_slot)

    def _release(self) -> None:
        self._mailbox._release_slot(self.address)

    def abandon(self) -> None:
        self._slot._notify_abandon()
        self._release()

    @property
    def _is_failed(self) -> bool:
        return self._failed

    def _mark_failed(self) -> None:
        self._failed = True

    @property
    def address(self) -> str:
        return self._slot._address