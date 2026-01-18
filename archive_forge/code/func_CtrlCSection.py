from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
def CtrlCSection(handler):
    """Run a section of code with CTRL-C redirected handler.

  Args:
    handler: func(), handler to call if SIGINT is received. In every case
      original Ctrl-C handler is not invoked.

  Returns:
    Context manager that redirects ctrl-c handler during its lifetime.
  """
    return _ReplaceSignal(signal.SIGINT, handler)