from __future__ import annotations
import sys
import eventlet
def _green_existing_locks():
    """Make locks created before monkey-patching safe.

    RLocks rely on a Lock and on Python 2, if an unpatched Lock blocks, it
    blocks the native thread. We need to replace these with green Locks.

    This was originally noticed in the stdlib logging module."""
    import gc
    import os
    import threading
    import eventlet.green.thread
    rlock_type = type(threading.RLock())
    tid = eventlet.green.thread.get_ident()

    def upgrade(old_lock):
        return _convert_py3_rlock(old_lock, tid)
    _upgrade_instances(sys.modules, rlock_type, upgrade)
    if 'PYTEST_CURRENT_TEST' in os.environ:
        return
    gc.collect()
    remaining_rlocks = len({o for o in gc.get_objects() if isinstance(o, rlock_type)})
    if remaining_rlocks:
        import logging
        logger = logging.Logger('eventlet')
        logger.error('{} RLock(s) were not greened,'.format(remaining_rlocks) + ' to fix this error make sure you run eventlet.monkey_patch() ' + 'before importing any other modules.')