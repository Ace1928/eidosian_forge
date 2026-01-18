from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import locale
import os
import struct
import sys
import six
from gslib.utils.constants import WINDOWS_1252
def IsRunningInCiEnvironment():
    """Returns True if running in a CI environment, e.g. GitHub CI."""
    on_github_ci = 'CI' in os.environ
    on_kokoro = 'KOKORO_ROOT' in os.environ
    return on_github_ci or on_kokoro