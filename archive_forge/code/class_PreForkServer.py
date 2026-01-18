import os
import sys
import time
import errno
import socket
import signal
import logging
import threading
import traceback
import email.message
import pyzor.config
import pyzor.account
import pyzor.engines.common
import pyzor.hacks.py26
class PreForkServer(Server):
    """The same as Server, but prefork itself when starting the self, by
    forking a number of child-processes.

    The parent process will then wait for all his child process to complete.
    """

    def __init__(self, address, database, passwd_fn, access_fn, prefork=4):
        """The same as Server.__init__ but requires a list of databases
        instead of a single database connection.
        """
        self.pids = None
        Server.__init__(self, address, database, passwd_fn, access_fn)
        self._prefork = prefork

    def serve_forever(self, poll_interval=0.5):
        """Fork the current process and wait for all children to finish."""
        pids = []
        for dummy in xrange(self._prefork):
            database = self.database.next()
            pid = os.fork()
            if not pid:
                self.database = database()
                Server.serve_forever(self, poll_interval=poll_interval)
                os._exit(0)
            else:
                pids.append(pid)
        self.pids = pids
        for pid in self.pids:
            _eintr_retry(os.waitpid, pid, 0)

    def shutdown(self):
        """If this is the parent process send the TERM signal to all children,
        else call the super method.
        """
        for pid in self.pids or ():
            os.kill(pid, signal.SIGTERM)
        if self.pids is None:
            Server.shutdown(self)

    def load_config(self):
        """If this is the parent process send the USR1 signal to all children,
        else call the super method.
        """
        for pid in self.pids or ():
            os.kill(pid, signal.SIGUSR1)
        if self.pids is None:
            Server.load_config(self)