import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
class PluggableReactorTests(TwistedModulesMixin, TestCase):
    """
    Tests for the reactor discovery/inspection APIs.
    """

    def setUp(self):
        """
        Override the L{reactors.getPlugins} function, normally bound to
        L{twisted.plugin.getPlugins}, in order to control which
        L{IReactorInstaller} plugins are seen as available.

        C{self.pluginResults} can be customized and will be used as the
        result of calls to C{reactors.getPlugins}.
        """
        self.pluginCalls = []
        self.pluginResults = []
        self.originalFunction = reactors.getPlugins
        reactors.getPlugins = self._getPlugins

    def tearDown(self):
        """
        Restore the original L{reactors.getPlugins}.
        """
        reactors.getPlugins = self.originalFunction

    def _getPlugins(self, interface, package=None):
        """
        Stand-in for the real getPlugins method which records its arguments
        and returns a fixed result.
        """
        self.pluginCalls.append((interface, package))
        return list(self.pluginResults)

    def test_getPluginReactorTypes(self):
        """
        Test that reactor plugins are returned from L{getReactorTypes}
        """
        name = 'fakereactortest'
        package = __name__ + '.fakereactor'
        description = 'description'
        self.pluginResults = [reactors.Reactor(name, package, description)]
        reactorTypes = reactors.getReactorTypes()
        self.assertEqual(self.pluginCalls, [(reactors.IReactorInstaller, None)])
        for r in reactorTypes:
            if r.shortName == name:
                self.assertEqual(r.description, description)
                break
        else:
            self.fail('Reactor plugin not present in getReactorTypes() result')

    def test_reactorInstallation(self):
        """
        Test that L{reactors.Reactor.install} loads the correct module and
        calls its install attribute.
        """
        installed = []

        def install():
            installed.append(True)
        fakeReactor = FakeReactor(install, 'fakereactortest', __name__, 'described')
        modules = {'fakereactortest': fakeReactor}
        self.replaceSysModules(modules)
        installer = reactors.Reactor('fakereactor', 'fakereactortest', 'described')
        installer.install()
        self.assertEqual(installed, [True])

    def test_installReactor(self):
        """
        Test that the L{reactors.installReactor} function correctly installs
        the specified reactor.
        """
        installed = []

        def install():
            installed.append(True)
        name = 'fakereactortest'
        package = __name__
        description = 'description'
        self.pluginResults = [FakeReactor(install, name, package, description)]
        reactors.installReactor(name)
        self.assertEqual(installed, [True])

    def test_installReactorReturnsReactor(self):
        """
        Test that the L{reactors.installReactor} function correctly returns
        the installed reactor.
        """
        reactor = object()

        def install():
            from twisted import internet
            self.patch(internet, 'reactor', reactor)
        name = 'fakereactortest'
        package = __name__
        description = 'description'
        self.pluginResults = [FakeReactor(install, name, package, description)]
        installed = reactors.installReactor(name)
        self.assertIs(installed, reactor)

    def test_installReactorMultiplePlugins(self):
        """
        Test that the L{reactors.installReactor} function correctly installs
        the specified reactor when there are multiple reactor plugins.
        """
        installed = []

        def install():
            installed.append(True)
        name = 'fakereactortest'
        package = __name__
        description = 'description'
        fakeReactor = FakeReactor(install, name, package, description)
        otherReactor = FakeReactor(lambda: None, 'otherreactor', package, description)
        self.pluginResults = [otherReactor, fakeReactor]
        reactors.installReactor(name)
        self.assertEqual(installed, [True])

    def test_installNonExistentReactor(self):
        """
        Test that L{reactors.installReactor} raises L{reactors.NoSuchReactor}
        when asked to install a reactor which it cannot find.
        """
        self.pluginResults = []
        self.assertRaises(reactors.NoSuchReactor, reactors.installReactor, 'somereactor')

    def test_installNotAvailableReactor(self):
        """
        Test that L{reactors.installReactor} raises an exception when asked to
        install a reactor which doesn't work in this environment.
        """

        def install():
            raise ImportError('Missing foo bar')
        name = 'fakereactortest'
        package = __name__
        description = 'description'
        self.pluginResults = [FakeReactor(install, name, package, description)]
        self.assertRaises(ImportError, reactors.installReactor, name)

    def test_reactorSelectionMixin(self):
        """
        Test that the reactor selected is installed as soon as possible, ie
        when the option is parsed.
        """
        executed = []
        INSTALL_EVENT = 'reactor installed'
        SUBCOMMAND_EVENT = 'subcommands loaded'

        class ReactorSelectionOptions(usage.Options, app.ReactorSelectionMixin):

            @property
            def subCommands(self):
                executed.append(SUBCOMMAND_EVENT)
                return [('subcommand', None, lambda: self, 'test subcommand')]

        def install():
            executed.append(INSTALL_EVENT)
        self.pluginResults = [FakeReactor(install, 'fakereactortest', __name__, 'described')]
        options = ReactorSelectionOptions()
        options.parseOptions(['--reactor', 'fakereactortest', 'subcommand'])
        self.assertEqual(executed[0], INSTALL_EVENT)
        self.assertEqual(executed.count(INSTALL_EVENT), 1)
        self.assertEqual(options['reactor'], 'fakereactortest')

    def test_reactorSelectionMixinNonExistent(self):
        """
        Test that the usage mixin exits when trying to use a non existent
        reactor (the name not matching to any reactor), giving an error
        message.
        """

        class ReactorSelectionOptions(usage.Options, app.ReactorSelectionMixin):
            pass
        self.pluginResults = []
        options = ReactorSelectionOptions()
        options.messageOutput = StringIO()
        e = self.assertRaises(usage.UsageError, options.parseOptions, ['--reactor', 'fakereactortest', 'subcommand'])
        self.assertIn('fakereactortest', e.args[0])
        self.assertIn('help-reactors', e.args[0])

    def test_reactorSelectionMixinNotAvailable(self):
        """
        Test that the usage mixin exits when trying to use a reactor not
        available (the reactor raises an error at installation), giving an
        error message.
        """

        class ReactorSelectionOptions(usage.Options, app.ReactorSelectionMixin):
            pass
        message = 'Missing foo bar'

        def install():
            raise ImportError(message)
        name = 'fakereactortest'
        package = __name__
        description = 'description'
        self.pluginResults = [FakeReactor(install, name, package, description)]
        options = ReactorSelectionOptions()
        options.messageOutput = StringIO()
        e = self.assertRaises(usage.UsageError, options.parseOptions, ['--reactor', 'fakereactortest', 'subcommand'])
        self.assertIn(message, e.args[0])
        self.assertIn('help-reactors', e.args[0])