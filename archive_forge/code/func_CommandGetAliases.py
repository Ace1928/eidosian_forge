from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def CommandGetAliases(self):
    """Get aliases for command.

    Returns:
      aliases: list of aliases for the command.
    """
    return self._command_aliases