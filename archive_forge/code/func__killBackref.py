import weakref
from pydispatch import saferef, robustapply, errors
def _killBackref(receiver, senderkey):
    """Do the actual removal of back reference from receiver to senderkey"""
    receiverkey = id(receiver)
    set = sendersBack.get(receiverkey, ())
    while senderkey in set:
        try:
            set.remove(senderkey)
        except:
            break
    if not set:
        try:
            del sendersBack[receiverkey]
        except KeyError:
            pass
    return True