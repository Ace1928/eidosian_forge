import argparse
import copy
import json
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _convert_extra_dhcp_options(parsed_args):
    dhcp_options = []
    for opt in parsed_args.extra_dhcp_options:
        option = {}
        option['opt_name'] = opt['name']
        if 'value' in opt:
            option['opt_value'] = opt['value']
        if 'ip-version' in opt:
            option['ip_version'] = opt['ip-version']
        dhcp_options.append(option)
    return dhcp_options