import weakref
from pydispatch import saferef, robustapply, errors
def _removeReceiver(receiver):
    """Remove receiver from connections."""
    if not sendersBack:
        return False
    backKey = id(receiver)
    try:
        backSet = sendersBack.pop(backKey)
    except KeyError:
        return False
    else:
        for senderkey in backSet:
            try:
                signals = list(connections[senderkey].keys())
            except KeyError:
                pass
            else:
                for signal in signals:
                    try:
                        receivers = connections[senderkey][signal]
                    except KeyError:
                        pass
                    else:
                        try:
                            receivers.remove(receiver)
                        except Exception:
                            pass
                    _cleanupConnections(senderkey, signal)