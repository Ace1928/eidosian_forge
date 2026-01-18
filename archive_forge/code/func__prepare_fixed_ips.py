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
def _prepare_fixed_ips(client_manager, parsed_args):
    """Fix and properly format fixed_ip option.

    Appropriately convert any subnet names to their respective ids.
    Convert fixed_ips in parsed args to be in valid dictionary format:
    {'subnet': 'foo'}.
    """
    client = client_manager.network
    ips = []
    if parsed_args.fixed_ip:
        for ip_spec in parsed_args.fixed_ip:
            if 'subnet' in ip_spec:
                subnet_name_id = ip_spec['subnet']
                if subnet_name_id:
                    _subnet = client.find_subnet(subnet_name_id, ignore_missing=False)
                    ip_spec['subnet_id'] = _subnet.id
                    del ip_spec['subnet']
            if 'ip-address' in ip_spec:
                ip_spec['ip_address'] = ip_spec['ip-address']
                del ip_spec['ip-address']
            ips.append(ip_spec)
    if ips:
        parsed_args.fixed_ip = ips