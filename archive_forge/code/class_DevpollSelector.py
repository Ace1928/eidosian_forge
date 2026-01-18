from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
class DevpollSelector(_PollLikeSelector):
    """Solaris /dev/poll selector."""
    _selector_cls = select.devpoll
    _EVENT_READ = select.POLLIN
    _EVENT_WRITE = select.POLLOUT

    def fileno(self):
        return self._selector.fileno()

    def close(self):
        self._selector.close()
        super().close()