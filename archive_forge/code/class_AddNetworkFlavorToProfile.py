import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class AddNetworkFlavorToProfile(command.Command):
    _description = _('Add a service profile to a network flavor')

    def get_parser(self, prog_name):
        parser = super(AddNetworkFlavorToProfile, self).get_parser(prog_name)
        parser.add_argument('flavor', metavar='<flavor>', help=_('Network flavor (name or ID)'))
        parser.add_argument('service_profile', metavar='<service-profile>', help=_('Service profile (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj_flavor = client.find_flavor(parsed_args.flavor, ignore_missing=False)
        obj_service_profile = client.find_service_profile(parsed_args.service_profile, ignore_missing=False)
        client.associate_flavor_with_service_profile(obj_flavor, obj_service_profile)