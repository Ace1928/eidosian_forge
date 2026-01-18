import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class CreateNetworkFlavor(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create new network flavor')

    def get_parser(self, prog_name):
        parser = super(CreateNetworkFlavor, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name for the flavor'))
        parser.add_argument('--service-type', metavar='<service-type>', required=True, help=_('Service type to which the flavor applies to: e.g. VPN (See openstack network service provider list for loaded examples.)'))
        parser.add_argument('--description', help=_('Description for the flavor'))
        parser.add_argument('--project', metavar='<project>', help=_("Owner's project (name or ID)"))
        identity_common.add_project_domain_option_to_parser(parser)
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable the flavor (default)'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable the flavor'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_flavor(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)