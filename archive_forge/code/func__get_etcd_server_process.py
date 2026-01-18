import atexit
import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Optional, TextIO, Union
def _get_etcd_server_process(self) -> subprocess.Popen:
    if not self._etcd_proc:
        raise RuntimeError('No etcd server process started. Call etcd_server.start() first')
    else:
        return self._etcd_proc