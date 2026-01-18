from __future__ import annotations
from typing import TYPE_CHECKING, Any
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED
from ._run import _NO_SEND, GLOBAL_RUN_CONTEXT, RunStatistics, Task
Block until there are no runnable tasks.

    This is useful in testing code when you want to give other tasks a
    chance to "settle down". The calling task is blocked, and doesn't wake
    up until all other tasks are also blocked for at least ``cushion``
    seconds. (Setting a non-zero ``cushion`` is intended to handle cases
    like two tasks talking to each other over a local socket, where we
    want to ignore the potential brief moment between a send and receive
    when all tasks are blocked.)

    Note that ``cushion`` is measured in *real* time, not the Trio clock
    time.

    If there are multiple tasks blocked in :func:`wait_all_tasks_blocked`,
    then the one with the shortest ``cushion`` is the one woken (and
    this task becoming unblocked resets the timers for the remaining
    tasks). If there are multiple tasks that have exactly the same
    ``cushion``, then all are woken.

    You should also consider :class:`trio.testing.Sequencer`, which
    provides a more explicit way to control execution ordering within a
    test, and will often produce more readable tests.

    Example:
      Here's an example of one way to test that Trio's locks are fair: we
      take the lock in the parent, start a child, wait for the child to be
      blocked waiting for the lock (!), and then check that we can't
      release and immediately re-acquire the lock::

         async def lock_taker(lock):
             await lock.acquire()
             lock.release()

         async def test_lock_fairness():
             lock = trio.Lock()
             await lock.acquire()
             async with trio.open_nursery() as nursery:
                 nursery.start_soon(lock_taker, lock)
                 # child hasn't run yet, we have the lock
                 assert lock.locked()
                 assert lock._owner is trio.lowlevel.current_task()
                 await trio.testing.wait_all_tasks_blocked()
                 # now the child has run and is blocked on lock.acquire(), we
                 # still have the lock
                 assert lock.locked()
                 assert lock._owner is trio.lowlevel.current_task()
                 lock.release()
                 try:
                     # The child has a prior claim, so we can't have it
                     lock.acquire_nowait()
                 except trio.WouldBlock:
                     assert lock._owner is not trio.lowlevel.current_task()
                     print("PASS")
                 else:
                     print("FAIL")

    