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
@staticmethod
def _Quote(command):
    """Quotes command in single quotes so it can be eval'd in coshell."""
    return "'{}'".format(command.replace("'", "'\\''"))