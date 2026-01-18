import gc
import inspect
import os
import pdb
import random
import sys
import time
import trace
import warnings
from typing import NoReturn, Optional, Type
from twisted import plugin
from twisted.application import app
from twisted.internet import defer
from twisted.python import failure, reflect, usage
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedModule
from twisted.trial import itrial, runner
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial.unittest import TestSuite
class _BasicOptions:
    """
    Basic options shared between trial and its local workers.
    """
    longdesc = 'trial loads and executes a suite of unit tests, obtained from modules, packages and files listed on the command line.'
    optFlags = [['help', 'h'], ['no-recurse', 'N', "Don't recurse into packages"], ['help-orders', None, 'Help on available test running orders'], ['help-reporters', None, 'Help on available output plugins (reporters)'], ['rterrors', 'e', 'realtime errors, print out tracebacks as soon as they occur'], ['unclean-warnings', None, 'Turn dirty reactor errors into warnings'], ['force-gc', None, 'Have Trial run gc.collect() before and after each test case.'], ['exitfirst', 'x', 'Exit after the first non-successful result (cannot be specified along with --jobs).']]
    optParameters = [['order', 'o', None, 'Specify what order to run test cases and methods. See --help-orders for more info.', _checkKnownRunOrder], ['random', 'z', None, 'Run tests in random order using the specified seed'], ['temp-directory', None, '_trial_temp', 'Path to use as working directory for tests.'], ['reporter', None, 'verbose', 'The reporter to use for this test run.  See --help-reporters for more info.']]
    compData = usage.Completions(optActions={'order': usage.CompleteList(_runOrders), 'reporter': _reporterAction, 'logfile': usage.CompleteFiles(descr='log file name'), 'random': usage.Completer(descr='random seed')}, extraActions=[usage.CompleteFiles('*.py', descr='file | module | package | TestCase | testMethod', repeat=True)])
    tracer: Optional[trace.Trace] = None

    def __init__(self):
        self['tests'] = []
        usage.Options.__init__(self)

    def getSynopsis(self):
        executableName = reflect.filenameToModuleName(sys.argv[0])
        if executableName.endswith('.__main__'):
            executableName = '{} -m {}'.format(os.path.basename(sys.executable), executableName.replace('.__main__', ''))
        return '{} [options] [[file|package|module|TestCase|testmethod]...]\n        '.format(executableName)

    def coverdir(self):
        """
        Return a L{FilePath} representing the directory into which coverage
        results should be written.
        """
        coverdir = 'coverage'
        result = FilePath(self['temp-directory']).child(coverdir)
        print(f'Setting coverage directory to {result.path}.')
        return result

    def opt_coverage(self):
        """
        Generate coverage information in the coverage file in the
        directory specified by the temp-directory option.
        """
        self.tracer = trace.Trace(count=1, trace=0)
        sys.settrace(self.tracer.globaltrace)
        self['coverage'] = True

    def opt_testmodule(self, filename):
        """
        Filename to grep for test cases (-*- test-case-name).
        """
        if not os.path.isfile(filename):
            sys.stderr.write(f"File {filename!r} doesn't exist\n")
            return
        filename = os.path.abspath(filename)
        if isTestFile(filename):
            self['tests'].append(filename)
        else:
            self['tests'].extend(getTestModules(filename))

    def opt_spew(self):
        """
        Print an insanely verbose log of everything that happens.  Useful
        when debugging freezes or locks in complex code.
        """
        from twisted.python.util import spewer
        sys.settrace(spewer)

    def opt_help_orders(self):
        synopsis = 'Trial can attempt to run test cases and their methods in a few different orders. You can select any of the following options using --order=<foo>.\n'
        print(synopsis)
        for name, (description, _) in sorted(_runOrders.items()):
            print('   ', name, '\t', description)
        sys.exit(0)

    def opt_help_reporters(self):
        synopsis = "Trial's output can be customized using plugins called Reporters. You can\nselect any of the following reporters using --reporter=<foo>\n"
        print(synopsis)
        for p in plugin.getPlugins(itrial.IReporter):
            print('   ', p.longOpt, '\t', p.description)
        sys.exit(0)

    def opt_disablegc(self):
        """
        Disable the garbage collector
        """
        self['disablegc'] = True
        gc.disable()

    def opt_tbformat(self, opt):
        """
        Specify the format to display tracebacks with. Valid formats are
        'plain', 'emacs', and 'cgitb' which uses the nicely verbose stdlib
        cgitb.text function
        """
        try:
            self['tbformat'] = TBFORMAT_MAP[opt]
        except KeyError:
            raise usage.UsageError("tbformat must be 'plain', 'emacs', or 'cgitb'.")

    def opt_recursionlimit(self, arg):
        """
        see sys.setrecursionlimit()
        """
        try:
            sys.setrecursionlimit(int(arg))
        except (TypeError, ValueError):
            raise usage.UsageError('argument to recursionlimit must be an integer')
        else:
            self['recursionlimit'] = int(arg)

    def opt_random(self, option):
        try:
            self['random'] = int(option)
        except ValueError:
            raise usage.UsageError('Argument to --random must be a positive integer')
        else:
            if self['random'] < 0:
                raise usage.UsageError('Argument to --random must be a positive integer')
            elif self['random'] == 0:
                self['random'] = int(time.time() * 100)

    def opt_without_module(self, option):
        """
        Fake the lack of the specified modules, separated with commas.
        """
        self['without-module'] = option
        for module in option.split(','):
            if module in sys.modules:
                warnings.warn("Module '%s' already imported, disabling anyway." % (module,), category=RuntimeWarning)
            sys.modules[module] = None

    def parseArgs(self, *args):
        self['tests'].extend(args)

    def _loadReporterByName(self, name):
        for p in plugin.getPlugins(itrial.IReporter):
            qual = f'{p.module}.{p.klass}'
            if p.longOpt == name:
                return reflect.namedAny(qual)
        raise usage.UsageError('Only pass names of Reporter plugins to --reporter. See --help-reporters for more info.')

    def postOptions(self):
        self['reporter'] = self._loadReporterByName(self['reporter'])
        if 'tbformat' not in self:
            self['tbformat'] = 'default'
        if self['order'] is not None and self['random'] is not None:
            raise usage.UsageError("You can't specify --random when using --order")