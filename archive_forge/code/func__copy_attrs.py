import os
import re
import subprocess
import sys
import time
import warnings
from . import QtCore, QtGui, QtWidgets, compat
from . import internals
def _copy_attrs(src, dst):
    for o in dir(src):
        if not hasattr(dst, o):
            setattr(dst, o, getattr(src, o))