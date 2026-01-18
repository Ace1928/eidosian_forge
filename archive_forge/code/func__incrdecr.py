from collections import deque
from twisted.internet.defer import Deferred, TimeoutError, fail
from twisted.protocols.basic import LineReceiver
from twisted.protocols.policies import TimeoutMixin
from twisted.python import log
from twisted.python.compat import nativeString, networkString
def _incrdecr(self, cmd, key, val):
    """
        Internal wrapper for incr/decr.
        """
    if self._disconnected:
        return fail(RuntimeError('not connected'))
    if not isinstance(key, bytes):
        return fail(ClientError(f'Invalid type for key: {type(key)}, expecting bytes'))
    if len(key) > self.MAX_KEY_LENGTH:
        return fail(ClientError('Key too long'))
    fullcmd = b' '.join([cmd, key, b'%d' % (int(val),)])
    self.sendLine(fullcmd)
    cmdObj = Command(cmd, key=key)
    self._current.append(cmdObj)
    return cmdObj._deferred