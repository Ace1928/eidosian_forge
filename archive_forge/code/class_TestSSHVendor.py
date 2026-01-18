import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
class TestSSHVendor:

    @staticmethod
    def run_command(host, command, username=None, port=None, password=None, key_filename=None):
        cmd, path = command.split(' ')
        cmd = cmd.split('-', 1)
        path = path.replace("'", '')
        p = subprocess.Popen([*cmd, path], bufsize=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return client.SubprocessWrapper(p)