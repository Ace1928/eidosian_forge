import py
import os
import sys
import marshal
class AutoFlush:

    def write(self, data):
        f.write(data)
        f.flush()

    def __getattr__(self, name):
        return getattr(f, name)