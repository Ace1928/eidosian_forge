import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
def _get_attrs_for_subports(client_manager, parsed_args):
    attrs = {}
    if 'set_subports' in parsed_args and parsed_args.set_subports is not None:
        attrs = _format_subports(client_manager, parsed_args.set_subports)
    if 'unset_subports' in parsed_args and parsed_args.unset_subports is not None:
        subports_list = []
        for subport in parsed_args.unset_subports:
            port_id = client_manager.network.find_port(subport)['id']
            subports_list.append({'port_id': port_id})
        attrs = subports_list
    return attrs