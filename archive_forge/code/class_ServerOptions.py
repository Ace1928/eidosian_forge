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
class ServerOptions(usage.Options, ReactorSelectionMixin):
    longdesc = 'twistd reads a twisted.application.service.Application out of a file and runs it.'
    optFlags = [['savestats', None, 'save the Stats object rather than the text output of the profiler.'], ['no_save', 'o', 'do not save state on shutdown'], ['encrypted', 'e', 'The specified tap/aos file is encrypted.']]
    optParameters = [['logfile', 'l', None, 'log to a specified file, - for stdout'], ['logger', None, None, 'A fully-qualified name to a log observer factory to use for the initial log observer.  Takes precedence over --logfile and --syslog (when available).'], ['profile', 'p', None, 'Run in profile mode, dumping results to specified file.'], ['profiler', None, 'cprofile', 'Name of the profiler to use (%s).' % ', '.join(AppProfiler.profilers)], ['file', 'f', 'twistd.tap', 'read the given .tap file'], ['python', 'y', None, 'read an application from within a Python file (implies -o)'], ['source', 's', None, 'Read an application from a .tas file (AOT format).'], ['rundir', 'd', '.', 'Change to a supplied directory before running']]
    compData = usage.Completions(mutuallyExclusive=[('file', 'python', 'source')], optActions={'file': usage.CompleteFiles('*.tap'), 'python': usage.CompleteFiles('*.(tac|py)'), 'source': usage.CompleteFiles('*.tas'), 'rundir': usage.CompleteDirs()})
    _getPlugins = staticmethod(plugin.getPlugins)

    def __init__(self, *a, **kw):
        self['debug'] = False
        if 'stdout' in kw:
            self.stdout = kw['stdout']
        else:
            self.stdout = sys.stdout
        usage.Options.__init__(self)

    def opt_debug(self):
        """
        Run the application in the Python Debugger (implies nodaemon),
        sending SIGUSR2 will drop into debugger
        """
        defer.setDebugging(True)
        failure.startDebugMode()
        self['debug'] = True
    opt_b = opt_debug

    def opt_spew(self):
        """
        Print an insanely verbose log of everything that happens.
        Useful when debugging freezes or locks in complex code.
        """
        sys.settrace(util.spewer)
        try:
            import threading
        except ImportError:
            return
        threading.settrace(util.spewer)

    def parseOptions(self, options=None):
        if options is None:
            options = sys.argv[1:] or ['--help']
        usage.Options.parseOptions(self, options)

    def postOptions(self):
        if self.subCommand or self['python']:
            self['no_save'] = True
        if self['logger'] is not None:
            try:
                self['logger'] = namedAny(self['logger'])
            except Exception as e:
                raise usage.UsageError("Logger '{}' could not be imported: {}".format(self['logger'], e))

    @property
    def subCommands(self):
        plugins = self._getPlugins(service.IServiceMaker)
        self.loadedPlugins = {}
        for plug in sorted(plugins, key=attrgetter('tapname')):
            self.loadedPlugins[plug.tapname] = plug
            yield (plug.tapname, None, lambda plug=plug: plug.options(), plug.description)