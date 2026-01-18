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
def MultithreadedMainSignalHandler(signal_num, cur_stack_frame):
    """Final signal handler for multi-threaded main process."""
    if signal_num == signal.SIGINT:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            stack_trace = ''.join(traceback.format_list(traceback.extract_stack()))
            err = 'DEBUG: Caught CTRL-C (signal %d) - Exception stack trace:\n    %s' % (signal_num, re.sub('\\n', '\n    ', stack_trace))
            try:
                sys.stderr.write(err.encode(UTF8))
            except (UnicodeDecodeError, TypeError) as e:
                sys.stderr.write(err)
        else:
            sys.stderr.write('Caught CTRL-C (signal %d) - exiting\n' % signal_num)
    metrics.LogFatalError(exception=ControlCException())
    metrics.Shutdown()
    KillProcess(os.getpid())