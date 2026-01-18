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
def _ProcessPluginPipes(self, section_name, proc, result, params, runtime_data):
    """Process the standard output and input streams of a plugin."""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        try:
            message = json.loads(line)
            self._ProcessMessage(proc.stdin, message, result, params, runtime_data)
        except ValueError:
            logging.info('%s: %s' % (section_name, line.rstrip()))