import atexit
import functools
import locale
import logging
import multiprocessing
import os
import traceback
import pathlib
import Pyro4.core
import argparse
from enum import IntEnum
import shutil
import socket
import struct
import collections
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
import uuid
import psutil
import Pyro4
from random import Random
from minerl.env import comms
import minerl.utils.process_watcher
def _launch_minecraft(self, port, headless, minecraft_dir, replaceable=False):
    """Launch Minecraft listening for malmoenv connections.
        Args:
            port:  the TCP port to listen on.
            installdir: the install dir name. Defaults to MalmoPlatform.
            Must be same as given (or defaulted) in download call if used.
            replaceable: whether or not to automatically restart Minecraft (default is false).
                         Does not work on Windows.
        Asserts:
            that the port specified is open.
        """
    launch_script = 'launchClient.sh'
    if os.name == 'nt':
        launch_script = 'launchClient.bat'
    launch_script = os.path.join(minecraft_dir, launch_script)
    rundir = os.path.join(minecraft_dir, 'run')
    cmd = [launch_script, '-port', str(port), '-env', '-runDir', rundir]
    if self.status_dir:
        cmd += ['-performanceDir', self.status_dir]
    if self._seed:
        cmd += ['-seed', ','.join([str(x) for x in self._seed])]
    if self._max_mem:
        cmd += ['-maxMem', self._max_mem]
    cmd_to_print = cmd[:] if not self._seed else cmd[:-2]
    self._logger.info('Starting Minecraft process: ' + str(cmd_to_print))
    if replaceable:
        cmd.append('-replaceable')
    preexec_fn = os.setsid if 'linux' in str(sys.platform) or sys.platform == 'darwin' else None
    minecraft_process = psutil.Popen(cmd, cwd=InstanceManager.MINECRAFT_DIR, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=preexec_fn)
    return minecraft_process