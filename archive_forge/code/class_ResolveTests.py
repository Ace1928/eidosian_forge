import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
@skipIf(not interfaces.IReactorProcess(reactor, None), "cannot run test: reactor doesn't support IReactorProcess")
class ResolveTests(TestCase):

    def testChildResolve(self):
        helperPath = os.path.abspath(self.mktemp())
        with open(helperPath, 'w') as helperFile:
            reactorName = reactor.__module__
            helperFile.write(resolve_helper % {'reactor': reactorName})
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        helperDeferred = Deferred()
        helperProto = ChildResolveProtocol(helperDeferred)
        reactor.spawnProcess(helperProto, sys.executable, ('python', '-u', helperPath), env)

        def cbFinished(result):
            reason, output, error = result
            output = b''.join(output)
            expected_output = b'done 127.0.0.1' + os.linesep.encode('ascii')
            if output != expected_output:
                self.fail('The child process failed to produce the desired results:\n   Reason for termination was: {!r}\n   Output stream was: {!r}\n   Error stream was: {!r}\n'.format(reason.getErrorMessage(), output, b''.join(error)))
        helperDeferred.addCallback(cbFinished)
        return helperDeferred