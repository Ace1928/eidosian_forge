import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
def _get_attrs(self, client_manager, parsed_args):
    attrs = {}
    if parsed_args.source_ip_address:
        attrs['source_ip_address'] = None
    if parsed_args.source_port:
        attrs['source_port'] = None
    if parsed_args.destination_ip_address:
        attrs['destination_ip_address'] = None
    if parsed_args.destination_port:
        attrs['destination_port'] = None
    if parsed_args.share:
        attrs['shared'] = False
    if parsed_args.enable_rule:
        attrs['enabled'] = False
    if parsed_args.source_firewall_group:
        attrs['source_firewall_group_id'] = None
    if parsed_args.source_firewall_group:
        attrs['destination_firewall_group_id'] = None
    return attrs