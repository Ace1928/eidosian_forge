import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class SetDomain(command.Command):
    _description = _('Set domain properties')

    def get_parser(self, prog_name):
        parser = super(SetDomain, self).get_parser(prog_name)
        parser.add_argument('domain', metavar='<domain>', help=_('Domain to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('New domain name'))
        parser.add_argument('--description', metavar='<description>', help=_('New domain description'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable domain'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable domain'))
        common.add_resource_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        domain = utils.find_resource(identity_client.domains, parsed_args.domain)
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = parsed_args.name
        if parsed_args.description:
            kwargs['description'] = parsed_args.description
        if parsed_args.enable:
            kwargs['enabled'] = True
        if parsed_args.disable:
            kwargs['enabled'] = False
        options = common.get_immutable_options(parsed_args)
        if options:
            kwargs['options'] = options
        identity_client.domains.update(domain.id, **kwargs)