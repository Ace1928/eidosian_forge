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
class UnrecognizedArgument(argparse.Action):

    def __init__(self, option_strings, dest, const=True, default=None, required=False, help=None, metavar=None, nargs=0):
        super(UnrecognizedArgument, self).__init__(option_strings=option_strings, dest=dest, nargs=nargs, const=const, default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error('unrecognized arguments: %s' % option_string)