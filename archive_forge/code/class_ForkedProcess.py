import atexit
import inspect
import multiprocessing.connection
import os
import signal
import subprocess
import sys
import time
import pickle
from ..Qt import QT_LIB, mkQApp
from ..util import cprint  # color printing for debugging
from .remoteproxy import (
import threading
class ForkedProcess(RemoteEventHandler):
    """
    ForkedProcess is a substitute for Process that uses os.fork() to generate a new process.
    This is much faster than starting a completely new interpreter and child processes
    automatically have a copy of the entire program state from before the fork. This
    makes it an appealing approach when parallelizing expensive computations. (see
    also Parallelizer)
    
    However, fork() comes with some caveats and limitations:

      - fork() is not available on Windows.
      - It is not possible to have a QApplication in both parent and child process
        (unless both QApplications are created _after_ the call to fork())
        Attempts by the forked process to access Qt GUI elements created by the parent
        will most likely cause the child to crash.
      - Likewise, database connections are unlikely to function correctly in a forked child.
      - Threads are not copied by fork(); the new process
        will have only one thread that starts wherever fork() was called in the parent process.
      - Forked processes are unceremoniously terminated when join() is called; they are not
        given any opportunity to clean up. (This prevents them calling any cleanup code that
        was only intended to be used by the parent process)
      - Normally when fork()ing, open file handles are shared with the parent process,
        which is potentially dangerous. ForkedProcess is careful to close all file handles
        that are not explicitly needed--stdout, stderr, and a single pipe to the parent
        process.
      
    """

    def __init__(self, name=None, target=0, preProxy=None, randomReseed=True):
        """
        When initializing, an optional target may be given. 
        If no target is specified, self.eventLoop will be used.
        If None is given, no target will be called (and it will be up 
        to the caller to properly shut down the forked process)
        
        preProxy may be a dict of values that will appear as ObjectProxy
        in the remote process (but do not need to be sent explicitly since 
        they are available immediately before the call to fork().
        Proxies will be availabe as self.proxies[name].
        
        If randomReseed is True, the built-in random and numpy.random generators
        will be reseeded in the child process.
        """
        self.hasJoined = False
        if target == 0:
            target = self.eventLoop
        if name is None:
            name = str(self)
        conn, remoteConn = multiprocessing.Pipe()
        proxyIDs = {}
        if preProxy is not None:
            for k, v in preProxy.items():
                proxyId = LocalObjectProxy.registerObject(v)
                proxyIDs[k] = proxyId
        ppid = os.getpid()
        pid = os.fork()
        if pid == 0:
            self.isParent = False
            os.setpgrp()
            conn.close()
            sys.stdin.close()
            fid = remoteConn.fileno()
            os.closerange(3, fid)
            os.closerange(fid + 1, 4096)

            def excepthook(*args):
                import traceback
                traceback.print_exception(*args)
            sys.excepthook = excepthook
            for qtlib in ('PyQt4', 'PySide', 'PyQt5'):
                if qtlib in sys.modules:
                    sys.modules[qtlib + '.QtGui'].QApplication = None
                    sys.modules.pop(qtlib + '.QtGui', None)
                    sys.modules.pop(qtlib + '.QtCore', None)
            atexit._exithandlers = []
            atexit.register(lambda: os._exit(0))
            if randomReseed:
                if 'numpy.random' in sys.modules:
                    sys.modules['numpy.random'].seed(os.getpid() ^ int(time.time() * 10000 % 10000))
                if 'random' in sys.modules:
                    sys.modules['random'].seed(os.getpid() ^ int(time.time() * 10000 % 10000))
            RemoteEventHandler.__init__(self, remoteConn, name + '_child', pid=ppid)
            self.forkedProxies = {}
            for name, proxyId in proxyIDs.items():
                self.forkedProxies[name] = ObjectProxy(ppid, proxyId=proxyId, typeStr=repr(preProxy[name]))
            if target is not None:
                target()
        else:
            self.isParent = True
            self.childPid = pid
            remoteConn.close()
            RemoteEventHandler.handlers = {}
            RemoteEventHandler.__init__(self, conn, name + '_parent', pid=pid)
            atexit.register(self.join)

    def eventLoop(self):
        while True:
            try:
                self.processRequests()
                time.sleep(0.01)
            except ClosedError:
                break
            except:
                print('Error occurred in forked event loop:')
                sys.excepthook(*sys.exc_info())
        sys.exit(0)

    def join(self, timeout=10):
        if self.hasJoined:
            return
        try:
            self.close(callSync='sync', timeout=timeout, noCleanup=True)
        except IOError:
            pass
        try:
            os.waitpid(self.childPid, 0)
        except OSError:
            pass
        self.conn.close()
        self.hasJoined = True

    def kill(self):
        """Immediately kill the forked remote process. 
        This is generally safe because forked processes are already
        expected to _avoid_ any cleanup at exit."""
        os.kill(self.childPid, signal.SIGKILL)
        self.hasJoined = True