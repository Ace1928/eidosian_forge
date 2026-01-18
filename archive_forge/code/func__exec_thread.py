import contextlib
import io
import logging
import os
import pwd
import shutil
import signal
import sys
import threading
import time
from unittest import mock
import fixtures
import testtools
from testtools import content
from oslo_rootwrap import client
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
from oslo_rootwrap.tests import run_daemon
def _exec_thread(self, fifo_path):
    try:
        self._thread_res = self.execute(['sh', '-c', 'echo > "%s"; sleep 1; echo OK' % fifo_path])
    except Exception as e:
        self._thread_res = e