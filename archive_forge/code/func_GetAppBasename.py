from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def GetAppBasename():
    """Returns the friendly basename of this application."""
    base = os.path.basename(sys.argv[0]).split('.')
    return base[0]