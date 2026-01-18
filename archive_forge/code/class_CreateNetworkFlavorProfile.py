import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class CreateNetworkFlavorProfile(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create new network flavor profile')

    def get_parser(self, prog_name):
        parser = super(CreateNetworkFlavorProfile, self).get_parser(prog_name)
        parser.add_argument('--project', metavar='<project>', help=_("Owner's project (name or ID)"))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--description', metavar='<description>', help=_('Description for the flavor profile'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable the flavor profile'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable the flavor profile'))
        parser.add_argument('--driver', help=_('Python module path to driver. This becomes required if --metainfo is missing and vice versa'))
        parser.add_argument('--metainfo', help=_('Metainfo for the flavor profile. This becomes required if --driver is missing and vice versa'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if parsed_args.driver is None and parsed_args.metainfo is None:
            msg = _('Either --driver or --metainfo or both are required')
            raise exceptions.CommandError(msg)
        obj = client.create_service_profile(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)