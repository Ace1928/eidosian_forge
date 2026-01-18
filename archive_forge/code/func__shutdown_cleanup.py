import os
import signal
import sys
import pickle
from .exceptions import RestartFreqExceeded
from time import monotonic
from io import BytesIO
def _shutdown_cleanup(signum, frame):
    if _should_have_exited[0]:
        os._exit(EX_SOFTWARE)
    maybe_setsignal(signum, signal.SIG_DFL)
    _should_have_exited[0] = True
    sys.exit(-(256 - signum))