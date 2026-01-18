import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
class FakeSshConnection(object):

    def __init__(self, rc, out=b'stdout', err=b'stderr'):
        self.rc = rc
        self.out = out
        self.err = err

    def exec_command(self, cmd, timeout=None):
        if timeout:
            raise socket.timeout()
        stdout = FakeSshStream(self.out)
        stdout.setup_channel(self.rc)
        return (io.BytesIO(), stdout, io.BytesIO(self.err))