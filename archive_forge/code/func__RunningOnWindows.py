from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
def _RunningOnWindows():
    """Lightweight mockable Windows check."""
    try:
        return bool(WindowsError)
    except NameError:
        return False