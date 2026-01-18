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
class AppProfiler:
    """
    Class which selects a specific profile runner based on configuration
    options.

    @ivar profiler: the name of the selected profiler.
    @type profiler: C{str}
    """
    profilers = {'profile': ProfileRunner, 'cprofile': CProfileRunner}

    def __init__(self, options):
        saveStats = options.get('savestats', False)
        profileOutput = options.get('profile', None)
        self.profiler = options.get('profiler', 'cprofile').lower()
        if self.profiler in self.profilers:
            profiler = self.profilers[self.profiler](profileOutput, saveStats)
            self.run = profiler.run
        else:
            raise SystemExit(f'Unsupported profiler name: {self.profiler}')