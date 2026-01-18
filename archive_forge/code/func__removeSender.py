import weakref
from pydispatch import saferef, robustapply, errors
def _removeSender(senderkey):
    """Remove senderkey from connections."""
    _removeBackrefs(senderkey)
    try:
        del connections[senderkey]
    except KeyError:
        pass
    try:
        del senders[senderkey]
    except:
        pass