import ctypes
import enum
import os
import platform
import sys
import numpy as np
class ErrorMessageCapture:

    def __init__(self):
        self.message = ''

    def report(self, x):
        self.message += x if isinstance(x, str) else x.decode('utf-8')