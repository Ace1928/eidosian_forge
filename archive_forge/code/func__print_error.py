from __future__ import (absolute_import, division, print_function)
import logging
import os.path
import subprocess
import sys
from configparser import ConfigParser
import ovirtsdk4 as sdk
from bcolors import bcolors
def _print_error(self, log):
    msg = 'Failed to generate var file.'
    log.error(msg)
    print('%s%s%s%s' % (FAIL, PREFIX, msg, END))