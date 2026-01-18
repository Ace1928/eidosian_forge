import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class SetIdentityProvider(command.Command):
    _description = _('Set identity provider properties')

    def get_parser(self, prog_name):
        parser = super(SetIdentityProvider, self).get_parser(prog_name)
        parser.add_argument('identity_provider', metavar='<identity-provider>', help=_('Identity provider to modify'))
        parser.add_argument('--description', metavar='<description>', help=_('Set identity provider description'))
        identity_remote_id_provider = parser.add_mutually_exclusive_group()
        identity_remote_id_provider.add_argument('--remote-id', metavar='<remote-id>', action='append', help=_('Remote IDs to associate with the Identity Provider (repeat option to provide multiple values)'))
        identity_remote_id_provider.add_argument('--remote-id-file', metavar='<file-name>', help=_('Name of a file that contains many remote IDs to associate with the identity provider, one per line'))
        parser.add_argument('--authorization-ttl', metavar='<authorization-ttl>', type=int, help=_('Time to keep the role assignments for users authenticating via this identity provider. Available since Identity API version 3.14 (Ussuri).'))
        enable_identity_provider = parser.add_mutually_exclusive_group()
        enable_identity_provider.add_argument('--enable', action='store_true', help=_('Enable the identity provider'))
        enable_identity_provider.add_argument('--disable', action='store_true', help=_('Disable the identity provider'))
        return parser

    def take_action(self, parsed_args):
        federation_client = self.app.client_manager.identity.federation
        if parsed_args.remote_id_file:
            file_content = utils.read_blob_file_contents(parsed_args.remote_id_file)
            remote_ids = file_content.splitlines()
            remote_ids = list(map(str.strip, remote_ids))
        elif parsed_args.remote_id:
            remote_ids = parsed_args.remote_id
        kwargs = {}
        if parsed_args.description:
            kwargs['description'] = parsed_args.description
        if parsed_args.enable:
            kwargs['enabled'] = True
        if parsed_args.disable:
            kwargs['enabled'] = False
        if parsed_args.remote_id_file or parsed_args.remote_id:
            kwargs['remote_ids'] = remote_ids
        auth_ttl = parsed_args.authorization_ttl
        if auth_ttl is not None:
            if auth_ttl < 0:
                msg = _('%(param)s must be positive integer or zero.') % {'param': 'authorization-ttl'}
                raise exceptions.CommandError(msg)
            kwargs['authorization_ttl'] = auth_ttl
        federation_client.identity_providers.update(parsed_args.identity_provider, **kwargs)