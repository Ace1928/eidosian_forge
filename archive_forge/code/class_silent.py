import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
class silent(object):
    """
    Silent sys.stderr at the system level
    """

    def __enter__(self):
        try:
            self.prevfd = os.dup(sys.stderr.fileno())
            os.close(sys.stderr.fileno())
        except io.UnsupportedOperation:
            self.prevfd = None
        self.prevstream = sys.stderr
        sys.stderr = open(os.devnull, 'r')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr.close()
        sys.stderr = self.prevstream
        if self.prevfd:
            os.dup2(self.prevfd, sys.stderr.fileno())
            os.close(self.prevfd)