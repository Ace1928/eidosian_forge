from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def GetCommandAliasList():
    """Return list of registered command aliases."""
    global _cmd_alias_list
    return _cmd_alias_list