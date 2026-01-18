import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class SetService(command.Command):
    _description = _('Set service properties')

    def get_parser(self, prog_name):
        parser = super(SetService, self).get_parser(prog_name)
        parser.add_argument('service', metavar='<service>', help=_('Service to modify (type, name or ID)'))
        parser.add_argument('--type', metavar='<type>', help=_('New service type (compute, image, identity, volume, etc)'))
        parser.add_argument('--name', metavar='<service-name>', help=_('New service name'))
        parser.add_argument('--description', metavar='<description>', help=_('New service description'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable service'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable service'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        service = common.find_service(identity_client, parsed_args.service)
        kwargs = {}
        if parsed_args.type:
            kwargs['type'] = parsed_args.type
        if parsed_args.name:
            kwargs['name'] = parsed_args.name
        if parsed_args.description:
            kwargs['description'] = parsed_args.description
        if parsed_args.enable:
            kwargs['enabled'] = True
        if parsed_args.disable:
            kwargs['enabled'] = False
        identity_client.services.update(service.id, **kwargs)