import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
def _format_subports(client_manager, subports):
    attrs = []
    for subport in subports:
        subport_attrs = {}
        if subport.get('port'):
            port_id = client_manager.network.find_port(subport['port'])['id']
            subport_attrs['port_id'] = port_id
        if subport.get('segmentation-id'):
            try:
                subport_attrs['segmentation_id'] = int(subport['segmentation-id'])
            except ValueError:
                msg = _("Segmentation-id '%s' is not an integer") % subport['segmentation-id']
                raise exceptions.CommandError(msg)
        if subport.get('segmentation-type'):
            subport_attrs['segmentation_type'] = subport['segmentation-type']
        attrs.append(subport_attrs)
    return attrs