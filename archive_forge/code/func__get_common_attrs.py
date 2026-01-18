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
def _get_common_attrs(client_manager, parsed_args, is_create=True):
    attrs = {}
    client = client_manager.network
    if is_create:
        if 'project' in parsed_args and parsed_args.project is not None:
            attrs['tenant_id'] = osc_utils.find_project(client_manager.identity, parsed_args.project, parsed_args.project_domain).id
    if parsed_args.name:
        attrs['name'] = str(parsed_args.name)
    if parsed_args.description:
        attrs['description'] = str(parsed_args.description)
    if parsed_args.protocol:
        protocol = parsed_args.protocol
        attrs['protocol'] = None if protocol == 'any' else protocol
    if parsed_args.action:
        attrs['action'] = parsed_args.action
    if parsed_args.ip_version:
        attrs['ip_version'] = str(parsed_args.ip_version)
    if parsed_args.source_port:
        attrs['source_port'] = parsed_args.source_port
    if parsed_args.no_source_port:
        attrs['source_port'] = None
    if parsed_args.source_ip_address:
        attrs['source_ip_address'] = parsed_args.source_ip_address
    if parsed_args.no_source_ip_address:
        attrs['source_ip_address'] = None
    if parsed_args.destination_port:
        attrs['destination_port'] = str(parsed_args.destination_port)
    if parsed_args.no_destination_port:
        attrs['destination_port'] = None
    if parsed_args.destination_ip_address:
        attrs['destination_ip_address'] = str(parsed_args.destination_ip_address)
    if parsed_args.no_destination_ip_address:
        attrs['destination_ip_address'] = None
    if parsed_args.enable_rule:
        attrs['enabled'] = True
    if parsed_args.disable_rule:
        attrs['enabled'] = False
    if parsed_args.share:
        attrs['shared'] = True
    if parsed_args.no_share:
        attrs['shared'] = False
    if parsed_args.source_firewall_group:
        attrs['source_firewall_group_id'] = client.find_firewall_group(parsed_args.source_firewall_group)['id']
    if parsed_args.no_source_firewall_group:
        attrs['source_firewall_group_id'] = None
    if parsed_args.destination_firewall_group:
        attrs['destination_firewall_group_id'] = client.find_firewall_group(parsed_args.destination_firewall_group)['id']
    if parsed_args.no_destination_firewall_group:
        attrs['destination_firewall_group_id'] = None
    return attrs