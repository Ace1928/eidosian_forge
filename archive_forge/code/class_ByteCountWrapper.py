import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
class ByteCountWrapper(object):
    """Wraps a file-like object, counting the number of bytes read."""

    def __init__(self, rfile):
        self.rfile = rfile
        self.bytes_read = 0

    def read(self, size=-1):
        data = self.rfile.read(size)
        self.bytes_read += len(data)
        return data

    def readline(self, size=-1):
        data = self.rfile.readline(size)
        self.bytes_read += len(data)
        return data

    def readlines(self, sizehint=0):
        total = 0
        lines = []
        line = self.readline()
        while line:
            lines.append(line)
            total += len(line)
            if 0 < sizehint <= total:
                break
            line = self.readline()
        return lines

    def close(self):
        self.rfile.close()

    def __iter__(self):
        return self

    def next(self):
        data = self.rfile.next()
        self.bytes_read += len(data)
        return data