import os
import pdb
import sys
import unittest as pyunit
from io import StringIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import plugin
from twisted.internet import defer
from twisted.plugins import twisted_trial
from twisted.python import failure, log, reflect
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.scripts import trial
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator
from twisted.trial.itrial import IReporter, ITestCase
class RunnerTests(unittest.SynchronousTestCase):

    def setUp(self):
        self.config = trial.Options()
        parts = reflect.qual(CapturingReporter).split('.')
        package = '.'.join(parts[:-1])
        klass = parts[-1]
        plugins = [twisted_trial._Reporter('Test Helper Reporter', package, description='Utility for unit testing.', longOpt='capturing', shortOpt=None, klass=klass)]

        def getPlugins(iface, *a, **kw):
            self.assertEqual(iface, IReporter)
            return plugins + list(self.original(iface, *a, **kw))
        self.original = plugin.getPlugins
        plugin.getPlugins = getPlugins
        self.standardReport = ['startTest', 'addSuccess', 'stopTest'] * 10

    def tearDown(self):
        plugin.getPlugins = self.original

    def parseOptions(self, args):
        self.config.parseOptions(args)

    def getRunner(self):
        r = trial._makeRunner(self.config)
        r.stream = StringIO()
        return r

    def test_runner_can_get_reporter(self):
        self.parseOptions([])
        result = self.config['reporter']
        runner = self.getRunner()
        self.assertEqual(result, runner._makeResult().__class__)

    def test_runner_get_result(self):
        self.parseOptions([])
        runner = self.getRunner()
        result = runner._makeResult()
        self.assertEqual(result.__class__, self.config['reporter'])

    def test_uncleanWarningsOffByDefault(self):
        """
        By default Trial sets the 'uncleanWarnings' option on the runner to
        False. This means that dirty reactor errors will be reported as
        errors. See L{test_reporter.DirtyReactorTests}.
        """
        self.parseOptions([])
        runner = self.getRunner()
        self.assertNotIsInstance(runner._makeResult(), reporter.UncleanWarningsReporterWrapper)

    def test_getsUncleanWarnings(self):
        """
        Specifying '--unclean-warnings' on the trial command line will cause
        reporters to be wrapped in a device which converts unclean errors to
        warnings.  See L{test_reporter.DirtyReactorTests} for implications.
        """
        self.parseOptions(['--unclean-warnings'])
        runner = self.getRunner()
        self.assertIsInstance(runner._makeResult(), reporter.UncleanWarningsReporterWrapper)

    def test_runner_working_directory(self):
        self.parseOptions(['--temp-directory', 'some_path'])
        runner = self.getRunner()
        self.assertEqual(runner.workingDirectory, 'some_path')

    def test_concurrentImplicitWorkingDirectory(self):
        """
        If no working directory is explicitly specified and the default
        working directory is in use by another runner, L{TrialRunner.run}
        selects a different default working directory to use.
        """
        self.parseOptions([])
        self.addCleanup(os.chdir, os.getcwd())
        runDirectory = FilePath(self.mktemp())
        runDirectory.makedirs()
        os.chdir(runDirectory.path)
        firstRunner = self.getRunner()
        secondRunner = self.getRunner()
        where = {}

        class ConcurrentCase(unittest.SynchronousTestCase):

            def test_first(self):
                """
                Start a second test run which will have a default working
                directory which is the same as the working directory of the
                test run already in progress.
                """
                where['concurrent'] = subsequentDirectory = os.getcwd()
                os.chdir(runDirectory.path)
                self.addCleanup(os.chdir, subsequentDirectory)
                secondRunner.run(ConcurrentCase('test_second'))

            def test_second(self):
                """
                Record the working directory for later analysis.
                """
                where['record'] = os.getcwd()
        result = firstRunner.run(ConcurrentCase('test_first'))
        bad = result.errors + result.failures
        if bad:
            self.fail(bad[0][1])
        self.assertEqual(where, {'concurrent': runDirectory.child('_trial_temp').path, 'record': runDirectory.child('_trial_temp-1').path})

    def test_concurrentExplicitWorkingDirectory(self):
        """
        If a working directory which is already in use is explicitly specified,
        L{TrialRunner.run} raises L{_WorkingDirectoryBusy}.
        """
        self.parseOptions(['--temp-directory', os.path.abspath(self.mktemp())])
        initialDirectory = os.getcwd()
        self.addCleanup(os.chdir, initialDirectory)
        firstRunner = self.getRunner()
        secondRunner = self.getRunner()

        class ConcurrentCase(unittest.SynchronousTestCase):

            def test_concurrent(self):
                """
                Try to start another runner in the same working directory and
                assert that it raises L{_WorkingDirectoryBusy}.
                """
                self.assertRaises(util._WorkingDirectoryBusy, secondRunner.run, ConcurrentCase('test_failure'))

            def test_failure(self):
                """
                Should not be called, always fails.
                """
                self.fail('test_failure should never be called.')
        result = firstRunner.run(ConcurrentCase('test_concurrent'))
        bad = result.errors + result.failures
        if bad:
            self.fail(bad[0][1])

    def test_runner_normal(self):
        self.parseOptions(['--temp-directory', self.mktemp(), '--reporter', 'capturing', 'twisted.trial.test.sample'])
        my_runner = self.getRunner()
        loader = runner.TestLoader()
        suite = loader.loadByName('twisted.trial.test.sample', True)
        result = my_runner.run(suite)
        self.assertEqual(self.standardReport, result._calls)

    def runSampleSuite(self, my_runner):
        loader = runner.TestLoader()
        suite = loader.loadByName('twisted.trial.test.sample', True)
        return my_runner.run(suite)

    def test_runnerDebug(self):
        """
        Trial uses its debugger if the `--debug` option is passed.
        """
        self.parseOptions(['--reporter', 'capturing', '--debug', 'twisted.trial.test.sample'])
        my_runner = self.getRunner()
        debugger = my_runner.debugger = CapturingDebugger()
        result = self.runSampleSuite(my_runner)
        self.assertEqual(self.standardReport, result._calls)
        self.assertEqual(['runcall'], debugger._calls)

    def test_runnerDebuggerDefaultsToPdb(self):
        """
        Trial uses pdb if no debugger is specified by `--debugger`
        """
        self.parseOptions(['--debug', 'twisted.trial.test.sample'])
        pdbrcFile = FilePath('pdbrc')
        pdbrcFile.touch()
        self.runcall_called = False

        def runcall(pdb, suite, result):
            self.runcall_called = True
        self.patch(pdb.Pdb, 'runcall', runcall)
        self.runSampleSuite(self.getRunner())
        self.assertTrue(self.runcall_called)

    def test_runnerDebuggerWithExplicitlyPassedPdb(self):
        """
        Trial uses pdb if pdb is passed explicitly to the `--debugger` arg.
        """
        self.parseOptions(['--reporter', 'capturing', '--debugger', 'pdb', '--debug', 'twisted.trial.test.sample'])
        self.runcall_called = False

        def runcall(pdb, suite, result):
            self.runcall_called = True
        self.patch(pdb.Pdb, 'runcall', runcall)
        self.runSampleSuite(self.getRunner())
        self.assertTrue(self.runcall_called)
    cdebugger = CapturingDebugger()

    def test_runnerDebugger(self):
        """
        Trial uses specified debugger if the debugger is available.
        """
        self.parseOptions(['--reporter', 'capturing', '--debugger', 'twisted.trial.test.test_runner.RunnerTests.cdebugger', '--debug', 'twisted.trial.test.sample'])
        my_runner = self.getRunner()
        result = self.runSampleSuite(my_runner)
        self.assertEqual(self.standardReport, result._calls)
        self.assertEqual(['runcall'], my_runner.debugger._calls)

    def test_exitfirst(self):
        """
        If trial was passed the C{--exitfirst} option, the constructed test
        result object is wrapped with L{reporter._ExitWrapper}.
        """
        self.parseOptions(['--exitfirst'])
        runner = self.getRunner()
        result = runner._makeResult()
        self.assertIsInstance(result, reporter._ExitWrapper)