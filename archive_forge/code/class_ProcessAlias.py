import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
@implementer(IAlias)
class ProcessAlias(AliasBase):
    """
    An alias which is handled by the execution of a program.

    @type path: L{list} of L{bytes}
    @ivar path: The arguments to pass to the process. The first string is
        the executable's name.

    @type program: L{bytes}
    @ivar program: The path of the program to be executed.

    @type reactor: L{IReactorTime <twisted.internet.interfaces.IReactorTime>}
        and L{IReactorProcess <twisted.internet.interfaces.IReactorProcess>}
        provider
    @ivar reactor: A reactor which will be used to create and timeout the
        child process.
    """
    reactor = reactor

    def __init__(self, path, *args):
        """
        @type path: L{bytes}
        @param path: The command to invoke the program consisting of the path
            to the executable followed by any arguments.

        @type args: 2-L{tuple} of (0) L{dict} mapping L{bytes} to L{IDomain}
            provider, (1) L{bytes}
        @param args: Arguments for L{AliasBase.__init__}.
        """
        AliasBase.__init__(self, *args)
        self.path = path.split()
        self.program = self.path[0]

    def __str__(self) -> str:
        """
        Build a string representation of this L{ProcessAlias} instance.

        @rtype: L{bytes}
        @return: A string containing the command used to invoke the process.
        """
        return f'<Process {self.path}>'

    def spawnProcess(self, proto, program, path):
        """
        Spawn a process.

        This wraps the L{spawnProcess
        <twisted.internet.interfaces.IReactorProcess.spawnProcess>} method on
        L{reactor} so that it can be customized for test purposes.

        @type proto: L{IProcessProtocol
            <twisted.internet.interfaces.IProcessProtocol>} provider
        @param proto: An object which will be notified of all events related to
            the created process.

        @type program: L{bytes}
        @param program: The full path name of the file to execute.

        @type path: L{list} of L{bytes}
        @param path: The arguments to pass to the process. The first string
            should be the executable's name.

        @rtype: L{IProcessTransport
            <twisted.internet.interfaces.IProcessTransport>} provider
        @return: A process transport.
        """
        return self.reactor.spawnProcess(proto, program, path)

    def createMessageReceiver(self):
        """
        Launch a process and create a message receiver to pass a message
        to the process.

        @rtype: L{MessageWrapper}
        @return: A message receiver which delivers a message to the process.
        """
        p = ProcessAliasProtocol()
        m = MessageWrapper(p, self.program, self.reactor)
        self.spawnProcess(p, self.program, self.path)
        return m