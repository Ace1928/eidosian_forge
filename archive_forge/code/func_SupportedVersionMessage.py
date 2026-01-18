from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
def SupportedVersionMessage(self):
    return 'Please use Python version {0}.{1} and up.'.format(PythonVersion.MIN_SUPPORTED_PY3_VERSION[0], PythonVersion.MIN_SUPPORTED_PY3_VERSION[1])