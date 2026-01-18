from __future__ import (absolute_import, division, print_function)
import logging
import os.path
import subprocess
from subprocess import call
import sys
import time
from configparser import ConfigParser
from bcolors import bcolors
def _log_to_file(self, log_file, command):
    with open(log_file, 'a') as f:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        for line in iter(proc.stdout.readline, ''):
            if 'TASK [' in line:
                print('\n%s%s%s\n' % (INFO, line, END))
            f.write(line)
        for line in iter(proc.stderr.readline, ''):
            f.write(line)
            print('%s%s%s' % (WARN, line, END))
    self._handle_result(command)