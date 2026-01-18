import weakref
from pydispatch import saferef, robustapply, errors
def _cleanupConnections(senderkey, signal):
    """Delete any empty signals for senderkey. Delete senderkey if empty."""
    try:
        receivers = connections[senderkey][signal]
    except:
        pass
    else:
        if not receivers:
            try:
                signals = connections[senderkey]
            except KeyError:
                pass
            else:
                del signals[signal]
                if not signals:
                    _removeSender(senderkey)