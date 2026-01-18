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
def _GetShellExecutable():
    """Gets the path to the Shell that should be used.

  First tries the current environment $SHELL, if set, then `bash` and `sh`. The
  first of these that is found is used.

  The shell must be Borne-compatible, as the commands that we execute with it
  are often bash/sh scripts.

  Returns:
    str, the path to the shell

  Raises:
    ValueError: if no Borne compatible shell is found
  """
    shells = ['/bin/bash', '/bin/sh']
    user_shell = encoding.GetEncodedValue(os.environ, 'SHELL')
    if user_shell and os.path.basename(user_shell) in _BORNE_COMPATIBLE_SHELLS:
        shells.insert(0, user_shell)
    for shell in shells:
        if os.path.isfile(shell):
            return shell
    raise ValueError("You must set your 'SHELL' environment variable to a valid Borne-compatible shell executable to use this tool.")