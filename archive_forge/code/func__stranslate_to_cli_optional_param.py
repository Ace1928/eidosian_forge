import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
@staticmethod
def _stranslate_to_cli_optional_param(param):
    if len(param) < 1 or not isinstance(param, str):
        raise exceptions.InvalidData('Provided wrong parameter for translation.')
    while not param[0:2] == '--':
        param = '-' + param
    return param.replace('_', '-')