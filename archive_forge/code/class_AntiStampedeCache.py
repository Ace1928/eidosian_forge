import datetime
import sys
import threading
import time
import cherrypy
from cherrypy.lib import cptools, httputil
class AntiStampedeCache(dict):
    """A storage system for cached items which reduces stampede collisions."""

    def wait(self, key, timeout=5, debug=False):
        """Return the cached value for the given key, or None.

        If timeout is not None, and the value is already
        being calculated by another thread, wait until the given timeout has
        elapsed. If the value is available before the timeout expires, it is
        returned. If not, None is returned, and a sentinel placed in the cache
        to signal other threads to wait.

        If timeout is None, no waiting is performed nor sentinels used.
        """
        value = self.get(key)
        if isinstance(value, threading.Event):
            if timeout is None:
                if debug:
                    cherrypy.log('No timeout', 'TOOLS.CACHING')
                return None
            if debug:
                cherrypy.log('Waiting up to %s seconds' % timeout, 'TOOLS.CACHING')
            value.wait(timeout)
            if value.result is not None:
                if debug:
                    cherrypy.log('Result!', 'TOOLS.CACHING')
                return value.result
            if debug:
                cherrypy.log('Timed out', 'TOOLS.CACHING')
            e = threading.Event()
            e.result = None
            dict.__setitem__(self, key, e)
            return None
        elif value is None:
            if debug:
                cherrypy.log('Timed out', 'TOOLS.CACHING')
            e = threading.Event()
            e.result = None
            dict.__setitem__(self, key, e)
        return value

    def __setitem__(self, key, value):
        """Set the cached value for the given key."""
        existing = self.get(key)
        dict.__setitem__(self, key, value)
        if isinstance(existing, threading.Event):
            existing.result = value
            existing.set()