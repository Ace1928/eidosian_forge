import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def _ignore_signals():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if signal.getsignal(signal.SIGQUIT) != signal.SIG_DFL:
        signal.signal(signal.SIGQUIT, signal.SIG_IGN)