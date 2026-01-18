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
class ApplicationRunner:
    """
    An object which helps running an application based on a config object.

    Subclass me and implement preApplication and postApplication
    methods. postApplication generally will want to run the reactor
    after starting the application.

    @ivar config: The config object, which provides a dict-like interface.

    @ivar application: Available in postApplication, but not
       preApplication. This is the application object.

    @ivar profilerFactory: Factory for creating a profiler object, able to
        profile the application if options are set accordingly.

    @ivar profiler: Instance provided by C{profilerFactory}.

    @ivar loggerFactory: Factory for creating object responsible for logging.

    @ivar logger: Instance provided by C{loggerFactory}.
    """
    profilerFactory = AppProfiler
    loggerFactory = AppLogger

    def __init__(self, config):
        self.config = config
        self.profiler = self.profilerFactory(config)
        self.logger = self.loggerFactory(config)

    def run(self):
        """
        Run the application.
        """
        self.preApplication()
        self.application = self.createOrGetApplication()
        self.logger.start(self.application)
        self.postApplication()
        self.logger.stop()

    def startReactor(self, reactor, oldstdout, oldstderr):
        """
        Run the reactor with the given configuration.  Subclasses should
        probably call this from C{postApplication}.

        @see: L{runReactorWithLogging}
        """
        if reactor is None:
            from twisted.internet import reactor
        runReactorWithLogging(self.config, oldstdout, oldstderr, self.profiler, reactor)
        if _ISupportsExitSignalCapturing.providedBy(reactor):
            self._exitSignal = reactor._exitSignal
        else:
            self._exitSignal = None

    def preApplication(self):
        """
        Override in subclass.

        This should set up any state necessary before loading and
        running the Application.
        """
        raise NotImplementedError()

    def postApplication(self):
        """
        Override in subclass.

        This will be called after the application has been loaded (so
        the C{application} attribute will be set). Generally this
        should start the application and run the reactor.
        """
        raise NotImplementedError()

    def createOrGetApplication(self):
        """
        Create or load an Application based on the parameters found in the
        given L{ServerOptions} instance.

        If a subcommand was used, the L{service.IServiceMaker} that it
        represents will be used to construct a service to be added to
        a newly-created Application.

        Otherwise, an application will be loaded based on parameters in
        the config.
        """
        if self.config.subCommand:
            plg = self.config.loadedPlugins[self.config.subCommand]
            ser = plg.makeService(self.config.subOptions)
            application = service.Application(plg.tapname)
            ser.setServiceParent(application)
        else:
            passphrase = getPassphrase(self.config['encrypted'])
            application = getApplication(self.config, passphrase)
        return application