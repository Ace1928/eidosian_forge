from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import signal
import sys
import traceback
from gslib import metrics
from gslib.exception import ControlCException
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
def ChildProcessSignalHandler(signal_num, cur_stack_frame):
    """Final signal handler for child processes."""
    KillProcess(os.getpid())