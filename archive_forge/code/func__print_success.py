from __future__ import (absolute_import, division, print_function)
import logging
import os.path
import subprocess
import sys
from configparser import ConfigParser
import ovirtsdk4 as sdk
from bcolors import bcolors
def _print_success(self, log):
    msg = 'Finished generating variable mapping file for oVirt ansible disaster recovery.'
    log.info(msg)
    print('%s%s%s%s' % (INFO, PREFIX, msg, END))