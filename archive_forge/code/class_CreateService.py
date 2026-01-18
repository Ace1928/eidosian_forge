import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateService(command.ShowOne):
    _description = _('Create new service')

    def get_parser(self, prog_name):
        parser = super(CreateService, self).get_parser(prog_name)
        parser.add_argument('type', metavar='<type>', help=_('New service type (compute, image, identity, volume, etc)'))
        parser.add_argument('--name', metavar='<name>', help=_('New service name'))
        parser.add_argument('--description', metavar='<description>', help=_('New service description'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        name = parsed_args.name
        type = parsed_args.type
        service = identity_client.services.create(name, type, parsed_args.description)
        info = {}
        info.update(service._info)
        return zip(*sorted(info.items()))