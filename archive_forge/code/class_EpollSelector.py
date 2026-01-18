from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
class EpollSelector(_PollLikeSelector):
    """Epoll-based selector."""
    _selector_cls = select.epoll
    _EVENT_READ = select.EPOLLIN
    _EVENT_WRITE = select.EPOLLOUT

    def fileno(self):
        return self._selector.fileno()

    def select(self, timeout=None):
        if timeout is None:
            timeout = -1
        elif timeout <= 0:
            timeout = 0
        else:
            timeout = math.ceil(timeout * 1000.0) * 0.001
        max_ev = max(len(self._fd_to_key), 1)
        ready = []
        try:
            fd_event_list = self._selector.poll(timeout, max_ev)
        except InterruptedError:
            return ready
        for fd, event in fd_event_list:
            events = 0
            if event & ~select.EPOLLIN:
                events |= EVENT_WRITE
            if event & ~select.EPOLLOUT:
                events |= EVENT_READ
            key = self._key_from_fd(fd)
            if key:
                ready.append((key, events & key.events))
        return ready

    def close(self):
        self._selector.close()
        super().close()