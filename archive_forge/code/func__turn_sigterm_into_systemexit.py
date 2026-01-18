import atexit
import errno
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from paste.deploy import loadapp, loadserver
from paste.script.command import Command, BadCommand
def _turn_sigterm_into_systemexit():
    """
    Attempts to turn a SIGTERM exception into a SystemExit exception.
    """
    try:
        import signal
    except ImportError:
        return

    def handle_term(signo, frame):
        raise SystemExit
    signal.signal(signal.SIGTERM, handle_term)