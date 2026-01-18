import errno
import logging
import os
import threading
import time
import six
from fasteners import _utils
def _try_acquire(self, blocking, watch):
    try:
        self.trylock()
    except IOError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN):
            if not blocking or watch.expired():
                return False
            else:
                raise _utils.RetryAgain()
        else:
            raise threading.ThreadError('Unable to acquire lock on `%(path)s` due to %(exception)s' % {'path': self.path, 'exception': e})
    else:
        return True