from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
class _ProcessHolder(object):
    """Process holder that can handle signals raised during processing."""

    def __init__(self):
        self.process = None
        self.signum = None

    def Handler(self, signum, unused_frame):
        """Handle the intercepted signal."""
        self.signum = signum
        if self.process:
            log.debug('Subprocess [{pid}] got [{signum}]'.format(signum=signum, pid=self.process.pid))
            if self.process.poll() is None:
                self.process.terminate()