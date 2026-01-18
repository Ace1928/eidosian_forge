import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def _IsInAppMain():
    """Returns True iff app.main or app.really_start is active."""
    f = sys._getframe().f_back
    app_dict = app.__dict__
    while f:
        if f.f_globals is app_dict and f.f_code.co_name in ('run', 'really_start'):
            return True
        f = f.f_back
    return False