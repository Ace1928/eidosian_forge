from __future__ import annotations
import threading
from collections import deque
from typing import TYPE_CHECKING, Callable, NoReturn, Tuple
import attrs
from .. import _core
from .._util import NoPublicConstructor, final
from ._wakeup_socketpair import WakeupSocketpair
@attrs.define
class EntryQueue:
    queue: deque[Job] = attrs.Factory(deque)
    idempotent_queue: dict[Job, None] = attrs.Factory(dict)
    wakeup: WakeupSocketpair = attrs.Factory(WakeupSocketpair)
    done: bool = False
    lock: threading.RLock = attrs.Factory(threading.RLock)

    async def task(self) -> None:
        assert _core.currently_ki_protected()
        assert self.lock.__class__.__module__ == '_thread'

        def run_cb(job: Job) -> None:
            sync_fn, args = job
            try:
                sync_fn(*args)
            except BaseException as exc:

                async def kill_everything(exc: BaseException) -> NoReturn:
                    raise exc
                try:
                    _core.spawn_system_task(kill_everything, exc)
                except RuntimeError:
                    parent_nursery = _core.current_task().parent_nursery
                    if parent_nursery is None:
                        raise AssertionError('Internal error: `parent_nursery` should never be `None`') from exc
                    parent_nursery.start_soon(kill_everything, exc)

        def run_all_bounded() -> None:
            for _ in range(len(self.queue)):
                run_cb(self.queue.popleft())
            for job in list(self.idempotent_queue):
                del self.idempotent_queue[job]
                run_cb(job)
        try:
            while True:
                run_all_bounded()
                if not self.queue and (not self.idempotent_queue):
                    await self.wakeup.wait_woken()
                else:
                    await _core.checkpoint()
        except _core.Cancelled:
            with self.lock:
                self.done = True
            run_all_bounded()
            assert not self.queue
            assert not self.idempotent_queue

    def close(self) -> None:
        self.wakeup.close()

    def size(self) -> int:
        return len(self.queue) + len(self.idempotent_queue)

    def run_sync_soon(self, sync_fn: Callable[[Unpack[PosArgsT]], object], *args: Unpack[PosArgsT], idempotent: bool=False) -> None:
        with self.lock:
            if self.done:
                raise _core.RunFinishedError('run() has exited')
            if idempotent:
                self.idempotent_queue[sync_fn, args] = None
            else:
                self.queue.append((sync_fn, args))
            self.wakeup.wakeup_thread_and_signal_safe()