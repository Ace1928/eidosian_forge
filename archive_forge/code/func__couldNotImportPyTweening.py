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
def _couldNotImportPyTweening(*unused_args, **unused_kwargs):
    """
        This function raises ``PyAutoGUIException``. It's used for the PyTweening function names if the PyTweening
        module failed to be imported.
        """
    raise PyAutoGUIException('PyAutoGUI was unable to import pytweening. Please install this module to enable the function you tried to call.')