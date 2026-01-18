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
def _couldNotImportPyMsgBox(*unused_args, **unused_kwargs):
    """
        This function raises ``PyAutoGUIException``. It's used for the PyMsgBox function names if the PyMsgbox module
        failed to be imported.
        """
    raise PyAutoGUIException('PyAutoGUI was unable to import pymsgbox. Please install this module to enable the function you tried to call.')