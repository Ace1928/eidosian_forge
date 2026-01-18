import os
import sys
import py
import tempfile
class DontReadFromInput:
    """Temporary stub class.  Ideally when stdin is accessed, the
    capturing should be turned off, with possibly all data captured
    so far sent to the screen.  This should be configurable, though,
    because in automated test runs it is better to crash than
    hang indefinitely.
    """

    def read(self, *args):
        raise IOError('reading from stdin while output is captured')
    readline = read
    readlines = read
    __iter__ = read

    def fileno(self):
        raise ValueError('redirected Stdin is pseudofile, has no fileno()')

    def isatty(self):
        return False

    def close(self):
        pass