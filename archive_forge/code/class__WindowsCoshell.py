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
class _WindowsCoshell(_CoshellBase):
    """The windows local coshell implementation.

  This implementation does not preserve shell coprocess state across Run().
  """

    def __init__(self):
        super(_WindowsCoshell, self).__init__(state_is_preserved=False)

    def Run(self, command, check_modes=False):
        """Runs command in the coshell and waits for it to complete."""
        del check_modes
        return subprocess.call(command, shell=True)

    def Interrupt(self):
        """Sends the interrupt signal to the coshell."""
        pass