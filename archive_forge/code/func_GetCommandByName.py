from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def GetCommandByName(name):
    """Get the command or None if name is not a registered command.

  Args:
    name:  name of command to look for

  Returns:
    Cmd instance holding the command or None
  """
    return GetCommandList().get(GetCommandAliasList().get(name))