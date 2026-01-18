import ast
import os
import platform
import re
import sys
from pyflakes import checker, __version__
from pyflakes import reporter as modReporter
def _exitOnSignal(sigName, message):
    """Handles a signal with sys.exit.

    Some of these signals (SIGPIPE, for example) don't exist or are invalid on
    Windows. So, ignore errors that might arise.
    """
    import signal
    try:
        sigNumber = getattr(signal, sigName)
    except AttributeError:
        return

    def handler(sig, f):
        sys.exit(message)
    try:
        signal.signal(sigNumber, handler)
    except ValueError:
        pass