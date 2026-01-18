from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
def _ProcessPluginStderr(self, section_name, stderr):
    """Process the standard error stream of a plugin.

    Standard error output is just written to the log at "warning" priority and
    otherwise ignored.

    Args:
      section_name: (str) Section name, to be attached to log messages.
      stderr: (file) Process standard error stream.
    """
    while True:
        line = stderr.readline()
        if not line:
            break
        logging.warn('%s: %s' % (section_name, line.rstrip()))