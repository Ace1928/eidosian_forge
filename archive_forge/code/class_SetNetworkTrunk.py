import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
class SetNetworkTrunk(command.Command):
    """Set network trunk properties"""

    def get_parser(self, prog_name):
        parser = super(SetNetworkTrunk, self).get_parser(prog_name)
        parser.add_argument('trunk', metavar='<trunk>', help=_('Trunk to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set trunk name'))
        parser.add_argument('--description', metavar='<description>', help=_('A description of the trunk'))
        parser.add_argument('--subport', metavar='<port=,segmentation-type=,segmentation-id=>', action=parseractions.MultiKeyValueAction, dest='set_subports', optional_keys=['segmentation-id', 'segmentation-type'], required_keys=['port'], help=_("Subport to add. Subport is of form 'port=<name or ID>,segmentation-type=<segmentation-type>,segmentation-id=<segmentation-ID>' (--subport) option can be repeated"))
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', help=_('Enable trunk'))
        admin_group.add_argument('--disable', action='store_true', help=_('Disable trunk'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        trunk_id = client.find_trunk(parsed_args.trunk)
        attrs = _get_attrs_for_trunk(self.app.client_manager, parsed_args)
        try:
            client.update_trunk(trunk_id, **attrs)
        except Exception as e:
            msg = _("Failed to set trunk '%(t)s': %(e)s") % {'t': parsed_args.trunk, 'e': e}
            raise exceptions.CommandError(msg)
        if parsed_args.set_subports:
            subport_attrs = _get_attrs_for_subports(self.app.client_manager, parsed_args)
            try:
                client.add_trunk_subports(trunk_id, subport_attrs)
            except Exception as e:
                msg = _("Failed to add subports to trunk '%(t)s': %(e)s") % {'t': parsed_args.trunk, 'e': e}
                raise exceptions.CommandError(msg)