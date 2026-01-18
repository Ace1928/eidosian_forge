import weakref
from pydispatch import saferef, robustapply, errors
def _removeBackrefs(senderkey):
    """Remove all back-references to this senderkey"""
    try:
        signals = connections[senderkey]
    except KeyError:
        signals = None
    else:
        items = signals.items()

        def allReceivers():
            for signal, set in items:
                for item in set:
                    yield item
        for receiver in allReceivers():
            _killBackref(receiver, senderkey)