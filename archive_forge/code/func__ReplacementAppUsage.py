from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def _ReplacementAppUsage(shorthelp=0, writeto_stdout=0, detailed_error=None, exitcode=None):
    AppcommandsUsage(shorthelp, writeto_stdout, detailed_error, exitcode=exitcode, show_cmd=None, show_global_flags=True)