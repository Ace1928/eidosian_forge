import contextlib
import errno
import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import FrozenSet, Optional
from urllib.error import URLError
from urllib.request import Request, quote, urljoin, urlopen
from jupyter_core.paths import jupyter_config_dir
from jupyter_server.extension.serverextension import GREEN_ENABLED, GREEN_OK, RED_DISABLED, RED_X
from jupyterlab_server.config import (
from jupyterlab_server.process import Process, WatchHelper, list2cmdline, which
from packaging.version import Version
from traitlets import Bool, HasTraits, Instance, List, Unicode, default
from jupyterlab._version import __version__
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.jlpmapp import HERE, YARN_PATH
from jupyterlab.semver import Range, gt, gte, lt, lte, make_semver
class ProgressProcess(Process):

    def __init__(self, cmd, logger=None, cwd=None, kill_event=None, env=None):
        """Start a subprocess that can be run asynchronously.

        Parameters
        ----------
        cmd: list
            The command to run.
        logger: :class:`~logger.Logger`, optional
            The logger instance.
        cwd: string, optional
            The cwd of the process.
        kill_event: :class:`~threading.Event`, optional
            An event used to kill the process operation.
        env: dict, optional
            The environment for the process.
        """
        if not isinstance(cmd, (list, tuple)):
            msg = 'Command must be given as a list'
            raise ValueError(msg)
        if kill_event and kill_event.is_set():
            msg = 'Process aborted'
            raise ValueError(msg)
        self.logger = _ensure_logger(logger)
        self._last_line = ''
        self.cmd = cmd
        self.logger.debug(f'> {list2cmdline(cmd)}')
        self.proc = self._create_process(cwd=cwd, env=env, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, universal_newlines=True, encoding='utf-8')
        self._kill_event = kill_event or Event()
        Process._procs.add(self)

    def wait(self):
        cache = []
        proc = self.proc
        kill_event = self._kill_event
        spinner = itertools.cycle(['-', '\\', '|', '/'])
        while proc.poll() is None:
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
            sys.stdout.write('\x08')
            if kill_event.is_set():
                self.terminate()
                msg = 'Process was aborted'
                raise ValueError(msg)
            try:
                out, _ = proc.communicate(timeout=0.1)
                cache.append(out)
            except subprocess.TimeoutExpired:
                continue
        self.logger.debug('\n'.join(cache))
        sys.stdout.flush()
        return self.terminate()