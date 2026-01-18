from __future__ import annotations
import atexit
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import weakref
from logging import Logger
from shutil import which as _which
from typing import Any
from tornado import gen
class WatchHelper(Process):
    """A process helper for a watch process."""

    def __init__(self, cmd: list[str], startup_regex: str, logger: Logger | None=None, cwd: str | None=None, kill_event: threading.Event | None=None, env: dict[str, str] | None=None) -> None:
        """Initialize the process helper.

        Parameters
        ----------
        cmd: list
            The command to run.
        startup_regex: string
            The regex to wait for at startup.
        logger: :class:`~logger.Logger`, optional
            The logger instance.
        cwd: string, optional
            The cwd of the process.
        env: dict, optional
            The environment for the process.
        kill_event: callable, optional
            A function to call to check if we should abort.
        """
        super().__init__(cmd, logger=logger, cwd=cwd, kill_event=kill_event, env=env)
        if pty is None:
            self._stdout = self.proc.stdout
        while 1:
            line = self._stdout.readline().decode('utf-8')
            if not line:
                msg = 'Process ended improperly'
                raise RuntimeError(msg)
            print(line.rstrip())
            if re.match(startup_regex, line):
                break
        self._read_thread = threading.Thread(target=self._read_incoming, daemon=True)
        self._read_thread.start()

    def terminate(self) -> int:
        """Terminate the process."""
        proc = self.proc
        if proc.poll() is None:
            if os.name != 'nt':
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait()
        finally:
            if self in Process._procs:
                Process._procs.remove(self)
        return proc.returncode

    def _read_incoming(self) -> None:
        """Run in a thread to read stdout and print"""
        fileno = self._stdout.fileno()
        while 1:
            try:
                buf = os.read(fileno, 1024)
            except OSError as e:
                self.logger.debug('Read incoming error %s', e)
                return
            if not buf:
                return
            print(buf.decode('utf-8'), end='')

    def _create_process(self, **kwargs: Any) -> subprocess.Popen[str]:
        """Create the watcher helper process."""
        kwargs['bufsize'] = 0
        if pty is not None:
            master, slave = pty.openpty()
            kwargs['stderr'] = kwargs['stdout'] = slave
            kwargs['start_new_session'] = True
            self._stdout = os.fdopen(master, 'rb')
        else:
            kwargs['stdout'] = subprocess.PIPE
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                kwargs['startupinfo'] = startupinfo
                kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
                kwargs['shell'] = True
        return super()._create_process(**kwargs)