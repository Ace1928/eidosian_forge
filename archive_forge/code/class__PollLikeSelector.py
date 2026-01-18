from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Mapping
import math
import select
import sys
class _PollLikeSelector(_BaseSelectorImpl):
    """Base class shared between poll, epoll and devpoll selectors."""
    _selector_cls = None
    _EVENT_READ = None
    _EVENT_WRITE = None

    def __init__(self):
        super().__init__()
        self._selector = self._selector_cls()

    def register(self, fileobj, events, data=None):
        key = super().register(fileobj, events, data)
        poller_events = 0
        if events & EVENT_READ:
            poller_events |= self._EVENT_READ
        if events & EVENT_WRITE:
            poller_events |= self._EVENT_WRITE
        try:
            self._selector.register(key.fd, poller_events)
        except:
            super().unregister(fileobj)
            raise
        return key

    def unregister(self, fileobj):
        key = super().unregister(fileobj)
        try:
            self._selector.unregister(key.fd)
        except OSError:
            pass
        return key

    def modify(self, fileobj, events, data=None):
        try:
            key = self._fd_to_key[self._fileobj_lookup(fileobj)]
        except KeyError:
            raise KeyError(f'{fileobj!r} is not registered') from None
        changed = False
        if events != key.events:
            selector_events = 0
            if events & EVENT_READ:
                selector_events |= self._EVENT_READ
            if events & EVENT_WRITE:
                selector_events |= self._EVENT_WRITE
            try:
                self._selector.modify(key.fd, selector_events)
            except:
                super().unregister(fileobj)
                raise
            changed = True
        if data != key.data:
            changed = True
        if changed:
            key = key._replace(events=events, data=data)
            self._fd_to_key[key.fd] = key
        return key

    def select(self, timeout=None):
        if timeout is None:
            timeout = None
        elif timeout <= 0:
            timeout = 0
        else:
            timeout = math.ceil(timeout * 1000.0)
        ready = []
        try:
            fd_event_list = self._selector.poll(timeout)
        except InterruptedError:
            return ready
        for fd, event in fd_event_list:
            events = 0
            if event & ~self._EVENT_READ:
                events |= EVENT_WRITE
            if event & ~self._EVENT_WRITE:
                events |= EVENT_READ
            key = self._key_from_fd(fd)
            if key:
                ready.append((key, events & key.events))
        return ready