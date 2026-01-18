from __future__ import (absolute_import, division, print_function)
import logging
import os.path
import subprocess
from subprocess import call
import sys
import time
from configparser import ConfigParser
from bcolors import bcolors
def _log_to_console(self, command, log):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for line in iter(proc.stdout.readline, ''):
        log.debug(line)
    for line in iter(proc.stderr.readline, ''):
        log.warn(line)