from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
class _BaseProcess(BaseProcess):
    """
    Base class for Process and PTYProcess.
    """
    status: Optional[int] = None
    pid = None

    def reapProcess(self):
        """
        Try to reap a process (without blocking) via waitpid.

        This is called when sigchild is caught or a Process object loses its
        "connection" (stdout is closed) This ought to result in reaping all
        zombie processes, since it will be called twice as often as it needs
        to be.

        (Unfortunately, this is a slightly experimental approach, since
        UNIX has no way to be really sure that your process is going to
        go away w/o blocking.  I don't want to block.)
        """
        try:
            try:
                pid, status = os.waitpid(self.pid, os.WNOHANG)
            except OSError as e:
                if e.errno == errno.ECHILD:
                    pid = None
                else:
                    raise
        except BaseException:
            log.msg(f'Failed to reap {self.pid}:')
            log.err()
            pid = None
        if pid:
            unregisterReapProcessHandler(pid, self)
            self.processEnded(status)

    def _getReason(self, status):
        exitCode = sig = None
        if os.WIFEXITED(status):
            exitCode = os.WEXITSTATUS(status)
        else:
            sig = os.WTERMSIG(status)
        if exitCode or sig:
            return error.ProcessTerminated(exitCode, sig, status)
        return error.ProcessDone(status)

    def signalProcess(self, signalID):
        """
        Send the given signal C{signalID} to the process. It'll translate a
        few signals ('HUP', 'STOP', 'INT', 'KILL', 'TERM') from a string
        representation to its int value, otherwise it'll pass directly the
        value provided

        @type signalID: C{str} or C{int}
        """
        if signalID in ('HUP', 'STOP', 'INT', 'KILL', 'TERM'):
            signalID = getattr(signal, f'SIG{signalID}')
        if self.pid is None:
            raise ProcessExitedAlready()
        try:
            os.kill(self.pid, signalID)
        except OSError as e:
            if e.errno == errno.ESRCH:
                raise ProcessExitedAlready()
            else:
                raise

    def _resetSignalDisposition(self):
        for signalnum in range(1, signal.NSIG):
            if signal.getsignal(signalnum) == signal.SIG_IGN:
                signal.signal(signalnum, signal.SIG_DFL)

    def _trySpawnInsteadOfFork(self, path, uid, gid, executable, args, environment, kwargs):
        """
        Try to use posix_spawnp() instead of fork(), if possible.

        This implementation returns False because the non-PTY subclass
        implements the actual logic; we can't yet use this for pty processes.

        @return: a boolean indicating whether posix_spawnp() was used or not.
        """
        return False

    def _fork(self, path, uid, gid, executable, args, environment, **kwargs):
        """
        Fork and then exec sub-process.

        @param path: the path where to run the new process.
        @type path: L{bytes} or L{unicode}

        @param uid: if defined, the uid used to run the new process.
        @type uid: L{int}

        @param gid: if defined, the gid used to run the new process.
        @type gid: L{int}

        @param executable: the executable to run in a new process.
        @type executable: L{str}

        @param args: arguments used to create the new process.
        @type args: L{list}.

        @param environment: environment used for the new process.
        @type environment: L{dict}.

        @param kwargs: keyword arguments to L{_setupChild} method.
        """
        if self._trySpawnInsteadOfFork(path, uid, gid, executable, args, environment, kwargs):
            return
        collectorEnabled = gc.isenabled()
        gc.disable()
        try:
            self.pid = os.fork()
        except BaseException:
            if collectorEnabled:
                gc.enable()
            raise
        else:
            if self.pid == 0:
                try:
                    sys.settrace(None)
                    self._setupChild(**kwargs)
                    self._execChild(path, uid, gid, executable, args, environment)
                except BaseException:
                    try:
                        stderr = io.TextIOWrapper(os.fdopen(2, 'wb'), encoding='utf-8')
                        msg = 'Upon execvpe {} {} in environment id {}\n:'.format(executable, str(args), id(environment))
                        stderr.write(msg)
                        traceback.print_exc(file=stderr)
                        stderr.flush()
                        for fd in range(3):
                            os.close(fd)
                    except BaseException:
                        pass
                os._exit(1)
        if collectorEnabled:
            gc.enable()
        self.status = -1

    def _setupChild(self, *args, **kwargs):
        """
        Setup the child process. Override in subclasses.
        """
        raise NotImplementedError()

    def _execChild(self, path, uid, gid, executable, args, environment):
        """
        The exec() which is done in the forked child.
        """
        if path:
            os.chdir(path)
        if uid is not None or gid is not None:
            if uid is None:
                uid = os.geteuid()
            if gid is None:
                gid = os.getegid()
            os.setuid(0)
            os.setgid(0)
            switchUID(uid, gid)
        os.execvpe(executable, args, environment)

    def __repr__(self) -> str:
        """
        String representation of a process.
        """
        return '<{} pid={} status={}>'.format(self.__class__.__name__, self.pid, self.status)