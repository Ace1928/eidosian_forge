import sys
import traceback
from pprint import pformat
from pathlib import Path
from IPython.core import ultratb
from IPython.core.release import author_email
from IPython.utils.sysinfo import sys_info
from IPython.utils.py3compat import input
from IPython.core.release import __version__ as version
from typing import Optional
Return a string containing a crash report.