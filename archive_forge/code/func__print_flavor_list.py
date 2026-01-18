import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def _print_flavor_list(cs, flavors, show_extra_specs=False):
    _translate_flavor_keys(flavors)
    headers = ['ID', 'Name', 'Memory_MiB', 'Disk', 'Ephemeral', 'Swap', 'VCPUs', 'RXTX_Factor', 'Is_Public']
    formatters = {}
    if show_extra_specs:
        if cs.api_version < api_versions.APIVersion('2.61'):
            formatters = {'extra_specs': _print_flavor_extra_specs}
        headers.append('extra_specs')
    if cs.api_version >= api_versions.APIVersion('2.55'):
        headers.append('Description')
    utils.print_list(flavors, headers, formatters)