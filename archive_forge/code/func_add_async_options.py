from __future__ import (absolute_import, division, print_function)
import copy
import operator
import argparse
import os
import os.path
import sys
import time
from jinja2 import __version__ as j2_version
import ansible
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.yaml import HAS_LIBYAML, yaml_load
from ansible.release import __version__
from ansible.utils.path import unfrackpath
def add_async_options(parser):
    """Add options for commands which can launch async tasks"""
    parser.add_argument('-P', '--poll', default=C.DEFAULT_POLL_INTERVAL, type=int, dest='poll_interval', help='set the poll interval if using -B (default=%s)' % C.DEFAULT_POLL_INTERVAL)
    parser.add_argument('-B', '--background', dest='seconds', type=int, default=0, help='run asynchronously, failing after X seconds (default=N/A)')