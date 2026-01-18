import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetNetworkFlavorProfile(common.NeutronCommandWithExtraArgs):
    _description = _('Set network flavor profile properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkFlavorProfile, self).get_parser(prog_name)
        parser.add_argument('flavor_profile', metavar='<flavor-profile>', help=_('Flavor profile to update (ID only)'))
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
        obj = client.find_service_profile(parsed_args.flavor_profile, ignore_missing=False)
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        client.update_service_profile(obj, **attrs)