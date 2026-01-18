import os
import struct
from twisted.internet import fdesc
from twisted.internet.abstract import FileDescriptor
from twisted.python import _inotify, log
def _addChildren(self, iwp):
    """
        This is a very private method, please don't even think about using it.

        Note that this is a fricking hack... it's because we cannot be fast
        enough in adding a watch to a directory and so we basically end up
        getting here too late if some operations have already been going on in
        the subdir, we basically need to catchup.  This eventually ends up
        meaning that we generate double events, your app must be resistant.
        """
    try:
        listdir = iwp.path.children()
    except OSError:
        return
    for f in listdir:
        if f.isdir():
            wd = self.watch(f, mask=iwp.mask, autoAdd=True, callbacks=iwp.callbacks)
            iwp._notify(f, IN_ISDIR | IN_CREATE)
            self.reactor.callLater(0, self._addChildren, self._watchpoints[wd])
        if f.isfile():
            iwp._notify(f, IN_CREATE | IN_CLOSE_WRITE)