import contextlib
import logging
import threading
import time
from tensorboard.util import tb_logging
@contextlib.contextmanager
def _log_latency(name, log_level):
    if not logger.isEnabledFor(log_level):
        yield
        return
    start_level = _store.nesting_level
    try:
        started = time.time()
        _store.nesting_level = start_level + 1
        indent = ' ' * 2 * start_level
        thread = threading.current_thread()
        prefix = '%s[%x]%s' % (thread.name, thread.ident, indent)
        _log(log_level, '%s ENTER %s', prefix, name)
        yield
    finally:
        _store.nesting_level = start_level
        elapsed = time.time() - started
        _log(log_level, '%s LEAVE %s - %0.6fs elapsed', prefix, name, elapsed)