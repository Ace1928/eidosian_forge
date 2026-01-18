import os
import sys
import time
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, threads
from twisted.python import failure, log, threadable, threadpool
from twisted.trial.unittest import TestCase
import time
import %(reactor)s
from twisted.internet import reactor
@skipIf(not interfaces.IReactorThreads(reactor, None), 'No thread support, nothing to test here.')
@skipIf(not interfaces.IReactorProcess(reactor, None), 'No process support, cannot run subprocess thread tests.')
class StartupBehaviorTests(TestCase):
    """
    Test cases for the behavior of the reactor threadpool near startup
    boundary conditions.

    In particular, this asserts that no threaded calls are attempted
    until the reactor starts up, that calls attempted before it starts
    are in fact executed once it has started, and that in both cases,
    the reactor properly cleans itself up (which is tested for
    somewhat implicitly, by requiring a child process be able to exit,
    something it cannot do unless the threadpool has been properly
    torn down).
    """

    def testCallBeforeStartupUnexecuted(self):
        progname = self.mktemp()
        with open(progname, 'w') as progfile:
            progfile.write(_callBeforeStartupProgram % {'reactor': reactor.__module__})

        def programFinished(result):
            out, err, reason = result
            if reason.check(error.ProcessTerminated):
                self.fail(f'Process did not exit cleanly (out: {out} err: {err})')
            if err:
                log.msg(f'Unexpected output on standard error: {err}')
            self.assertFalse(out, f'Expected no output, instead received:\n{out}')

        def programTimeout(err):
            err.trap(error.TimeoutError)
            proto.signalProcess('KILL')
            return err
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        d = defer.Deferred().addCallbacks(programFinished, programTimeout)
        proto = ThreadStartupProcessProtocol(d)
        reactor.spawnProcess(proto, sys.executable, ('python', progname), env)
        return d