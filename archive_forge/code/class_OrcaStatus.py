import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import warnings
from copy import copy
from contextlib import contextmanager
from pathlib import Path
from shutil import which
import tenacity
import plotly
from plotly.files import PLOTLY_DIR, ensure_writable_plotly_dir
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.optional_imports import get_module
class OrcaStatus(object):
    """
    Class to store information about the current status of the orca server.
    """
    _props = {'state': 'unvalidated', 'executable_list': None, 'version': None, 'pid': None, 'port': None, 'command': None}

    @property
    def state(self):
        """
        A string representing the state of the orca server process

        One of:
          - unvalidated: The orca executable has not yet been searched for or
            tested to make sure its valid.
          - validated: The orca executable has been located and tested for
            validity, but it is not running.
          - running: The orca server process is currently running.
        """
        return self._props['state']

    @property
    def executable(self):
        """
        If the `state` property is 'validated' or 'running', this property
        contains the full path to the orca executable.

        This path can be specified explicitly by setting the `executable`
        property of the `plotly.io.orca.config` object.

        This property will be None if the `state` is 'unvalidated'.
        """
        executable_list = self._props['executable_list']
        if executable_list is None:
            return None
        else:
            return ' '.join(executable_list)

    @property
    def version(self):
        """
        If the `state` property is 'validated' or 'running', this property
        contains the version of the validated orca executable.

        This property will be None if the `state` is 'unvalidated'.
        """
        return self._props['version']

    @property
    def pid(self):
        """
        The process id of the orca server process, if any. This property
        will be None if the `state` is not 'running'.
        """
        return self._props['pid']

    @property
    def port(self):
        """
        The port number that the orca server process is listening to, if any.
        This property will be None if the `state` is not 'running'.

        This port can be specified explicitly by setting the `port`
        property of the `plotly.io.orca.config` object.
        """
        return self._props['port']

    @property
    def command(self):
        """
        The command arguments used to launch the running orca server, if any.
        This property will be None if the `state` is not 'running'.
        """
        return self._props['command']

    def __repr__(self):
        """
        Display a nice representation of the current orca server status.
        """
        return 'orca status\n-----------\n    state: {state}\n    executable: {executable}\n    version: {version}\n    port: {port}\n    pid: {pid}\n    command: {command}\n\n'.format(executable=self.executable, version=self.version, port=self.port, pid=self.pid, state=self.state, command=self.command)