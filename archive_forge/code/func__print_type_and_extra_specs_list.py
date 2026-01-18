from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
def _print_type_and_extra_specs_list(stypes, columns=None):
    """Prints extra specs for a list of share types or share group types."""
    formatters = {'all_extra_specs': _print_type_extra_specs}
    fields = ['ID', 'Name', 'all_extra_specs']
    if columns is not None:
        fields = _split_columns(columns=columns, title=False)
    cliutils.print_list(stypes, fields, formatters)