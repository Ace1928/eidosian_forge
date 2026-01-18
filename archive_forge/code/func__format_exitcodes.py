import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
def _format_exitcodes(exitcodes):
    """Format a list of exit code with names of the signals if possible"""
    str_exitcodes = [f'{_get_exitcode_name(e)}({e})' for e in exitcodes if e is not None]
    return '{' + ', '.join(str_exitcodes) + '}'