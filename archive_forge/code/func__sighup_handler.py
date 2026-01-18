import signal
import weakref
from ... import trace
def _sighup_handler(signal_number, interrupted_frame):
    """This is the actual function that is registered for handling SIGHUP.

    It will call out to all the registered functions, letting them know that a
    graceful termination has been requested.
    """
    if _on_sighup is None:
        return
    trace.mutter('Caught SIGHUP, sending graceful shutdown requests.')
    for ref in _on_sighup.valuerefs():
        try:
            cb = ref()
            if cb is not None:
                cb()
        except KeyboardInterrupt:
            raise
        except Exception:
            trace.mutter('Error occurred while running SIGHUP handlers:')
            trace.log_exception_quietly()