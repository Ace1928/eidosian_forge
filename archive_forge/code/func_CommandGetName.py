from mx import DateTime
from __future__ import absolute_import
from __future__ import print_function
import os
import pdb
import sys
import traceback
from absl import app
from absl import flags
def CommandGetName(self):
    """Get name of command.

    Returns:
      Command name.
    """
    return self._command_name