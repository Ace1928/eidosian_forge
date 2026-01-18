import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class SetSfcPortPair(command.Command):
    _description = _('Set port pair properties')

    def get_parser(self, prog_name):
        parser = super(SetSfcPortPair, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help=_('Name of the port pair'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the port pair'))
        parser.add_argument('port_pair', metavar='<port-pair>', help=_('Port pair to modify (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        port_pair_id = client.find_sfc_port_pair(parsed_args.port_pair, ignore_missing=False)['id']
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        try:
            client.update_sfc_port_pair(port_pair_id, **attrs)
        except Exception as e:
            msg = _("Failed to update port pair '%(port_pair)s': %(e)s") % {'port_pair': parsed_args.port_pair, 'e': e}
            raise exceptions.CommandError(msg)