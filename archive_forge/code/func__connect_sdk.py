from __future__ import (absolute_import, division, print_function)
import logging
import os.path
import subprocess
import sys
from configparser import ConfigParser
import ovirtsdk4 as sdk
from bcolors import bcolors
def _connect_sdk(self, url, username, password, ca):
    connection = sdk.Connection(url=url, username=username, password=password, ca_file=ca)
    return connection