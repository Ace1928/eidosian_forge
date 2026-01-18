import errno
import fcntl
import os
import subprocess
import time
from . import Connection, ConnectionException
class XvfbTest:
    """ A helper class for testing things with nosetests. This class will run
    each test in its own fresh xvfb, leaving you with an xcffib connection to
    that X session as `self.conn` for use in testing. """
    xtrace = False

    def __init__(self, width=800, height=600, depth=16):
        self.width = width
        self.height = height
        self.depth = depth

    def spawn(self, cmd):
        """ Spawn a command but swallow its output. """
        return subprocess.Popen(cmd)

    def _restore_display(self):
        if self._old_display is None:
            del os.environ['DISPLAY']
        else:
            os.environ['DISPLAY'] = self._old_display

    def setUp(self):
        self._old_display = os.environ.get('DISPLAY')
        self._display, self._display_lock = find_display()
        os.environ['DISPLAY'] = ':%d' % self._display
        self._xvfb = self.spawn(self._xvfb_command())
        if self.xtrace:
            subprocess.Popen(['xtrace', '-n'])
            os.environ['DISPLAY'] = ':9'
        try:
            self.conn = self._connect_to_xvfb()
        except AssertionError:
            self._restore_display()
            raise

    def tearDown(self):
        try:
            self.conn.disconnect()
        except ConnectionException:
            pass
        finally:
            self.conn = None
        self._xvfb.kill()
        self._xvfb.wait()
        self._xvfb = None
        try:
            self._display_lock.close()
            os.remove(lock_path(self._display))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        finally:
            self._restore_display()

    def __enter__(self):
        self.setUp()
        return self

    def __exit__(self, type, value, traceback):
        self.tearDown()

    def _xvfb_command(self):
        """ You can override this if you have some extra args for Xvfb or
        whatever. At this point, os.environ['DISPLAY'] is set to something Xvfb
        can use. """
        screen = '%sx%sx%s' % (self.width, self.height, self.depth)
        return ['Xvfb', os.environ['DISPLAY'], '-screen', '0', screen]

    def _connect_to_xvfb(self):
        for _ in range(100):
            try:
                conn = Connection(os.environ['DISPLAY'])
                conn.invalid()
                return conn
            except ConnectionException:
                time.sleep(0.2)
        assert False, "couldn't connect to xvfb"