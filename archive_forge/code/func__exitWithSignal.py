import getpass
import os
import pdb
import signal
import sys
import traceback
import warnings
from operator import attrgetter
from twisted import copyright, logger, plugin
from twisted.application import reactors, service
from twisted.application.reactors import NoSuchReactor, installReactor
from twisted.internet import defer
from twisted.internet.interfaces import _ISupportsExitSignalCapturing
from twisted.persisted import sob
from twisted.python import failure, log, logfile, runtime, usage, util
from twisted.python.reflect import namedAny, namedModule, qual
def _exitWithSignal(sig):
    """
    Force the application to terminate with the specified signal by replacing
    the signal handler with the default and sending the signal to ourselves.

    @param sig:  Signal to use to terminate the process with C{os.kill}.
    @type sig:  C{int}
    """
    signal.signal(sig, signal.SIG_DFL)
    os.kill(os.getpid(), sig)