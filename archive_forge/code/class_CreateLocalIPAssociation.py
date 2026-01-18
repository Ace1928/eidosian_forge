import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class CreateLocalIPAssociation(command.ShowOne):
    _description = _('Create Local IP Association')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('local_ip', metavar='<local-ip>', help=_('Local IP that the port association belongs to (Name or ID)'))
        parser.add_argument('fixed_port', metavar='<fixed-port>', help=_('The ID or Name of Port to allocate Local IP Association'))
        parser.add_argument('--fixed-ip', metavar='<fixed-ip>', help=_('Fixed IP for Local IP Association'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = {}
        port = client.find_port(parsed_args.fixed_port, ignore_missing=False)
        attrs['fixed_port_id'] = port.id
        if parsed_args.fixed_ip:
            attrs['fixed_ip'] = parsed_args.fixed_ip
        local_ip = client.find_local_ip(parsed_args.local_ip, ignore_missing=False)
        obj = client.create_local_ip_association(local_ip.id, **attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)