from __future__ import unicode_literals, absolute_import
import sys
import abc
import errno
import select
import six
class Python3Selector(Selector):
    """
    Use of the Python3 'selectors' module.

    NOTE: Only use on Python 3.5 or newer!
    """

    def __init__(self):
        assert sys.version_info >= (3, 5)
        import selectors
        self._sel = selectors.DefaultSelector()

    def register(self, fd):
        assert isinstance(fd, int)
        import selectors
        self._sel.register(fd, selectors.EVENT_READ, None)

    def unregister(self, fd):
        assert isinstance(fd, int)
        self._sel.unregister(fd)

    def select(self, timeout):
        events = self._sel.select(timeout=timeout)
        return [key.fileobj for key, mask in events]

    def close(self):
        self._sel.close()