from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
class ImageNotFoundException(PyAutoGUIException):
    """
    This exception is the PyAutoGUI version of PyScreeze's `ImageNotFoundException`, which is raised when a locate*()
    function call is unable to find an image.

    Ideally, `pyscreeze.ImageNotFoundException` should never be raised by PyAutoGUI.
    """