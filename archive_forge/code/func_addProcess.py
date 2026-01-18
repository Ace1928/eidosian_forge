from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
def addProcess(self, name, args, uid=None, gid=None, env={}, cwd=None):
    """
        Add a new monitored process and start it immediately if the
        L{ProcessMonitor} service is running.

        Note that args are passed to the system call, not to the shell. If
        running the shell is desired, the common idiom is to use
        C{ProcessMonitor.addProcess("name", ['/bin/sh', '-c', shell_script])}

        @param name: A name for this process.  This value must be
            unique across all processes added to this monitor.
        @type name: C{str}
        @param args: The argv sequence for the process to launch.
        @param uid: The user ID to use to run the process.  If L{None},
            the current UID is used.
        @type uid: C{int}
        @param gid: The group ID to use to run the process.  If L{None},
            the current GID is used.
        @type uid: C{int}
        @param env: The environment to give to the launched process. See
            L{IReactorProcess.spawnProcess}'s C{env} parameter.
        @type env: C{dict}
        @param cwd: The initial working directory of the launched process.
            The default of C{None} means inheriting the laucnhing process's
            working directory.
        @type env: C{dict}
        @raise KeyError: If a process with the given name already exists.
        """
    if name in self._processes:
        raise KeyError(f'remove {name} first')
    self._processes[name] = _Process(args, uid, gid, env, cwd)
    self.delay[name] = self.minRestartDelay
    if self.running:
        self.startProcess(name)