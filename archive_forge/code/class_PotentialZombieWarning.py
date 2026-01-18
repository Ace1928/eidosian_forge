import socket
from incremental import Version
from twisted.python import deprecate
class PotentialZombieWarning(Warning):
    """
    Emitted when L{IReactorProcess.spawnProcess} is called in a way which may
    result in termination of the created child process not being reported.

    Deprecated in Twisted 10.0.
    """
    MESSAGE = 'spawnProcess called, but the SIGCHLD handler is not installed. This probably means you have not yet called reactor.run, or called reactor.run(installSignalHandler=0). You will probably never see this process finish, and it may become a zombie process.'