import re
import sys
import warnings
class WriteWrapper:

    def __init__(self, wsgi_writer):
        self.writer = wsgi_writer

    def __call__(self, s):
        assert_(type(s) is bytes)
        self.writer(s)