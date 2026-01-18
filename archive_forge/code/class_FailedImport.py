import os
import re
import subprocess
import sys
import time
import warnings
from . import QtCore, QtGui, QtWidgets, compat
from . import internals
class FailedImport(object):
    """Used to defer ImportErrors until we are sure the module is needed.
    """

    def __init__(self, err):
        self.err = err

    def __getattr__(self, attr):
        raise self.err