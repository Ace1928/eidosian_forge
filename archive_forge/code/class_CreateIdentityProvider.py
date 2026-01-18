import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateIdentityProvider(command.ShowOne):
    _description = _('Create new identity provider')

    def get_parser(self, prog_name):
        parser = super(CreateIdentityProvider, self).get_parser(prog_name)
        parser.add_argument('identity_provider_id', metavar='<name>', help=_('New identity provider name (must be unique)'))
        identity_remote_id_provider = parser.add_mutually_exclusive_group()
        identity_remote_id_provider.add_argument('--remote-id', metavar='<remote-id>', action='append', help=_('Remote IDs to associate with the Identity Provider (repeat option to provide multiple values)'))
        identity_remote_id_provider.add_argument('--remote-id-file', metavar='<file-name>', help=_('Name of a file that contains many remote IDs to associate with the identity provider, one per line'))
        parser.add_argument('--description', metavar='<description>', help=_('New identity provider description'))
        parser.add_argument('--domain', metavar='<domain>', help=_('Domain to associate with the identity provider. If not specified, a domain will be created automatically. (Name or ID)'))
        parser.add_argument('--authorization-ttl', metavar='<authorization-ttl>', type=int, help=_('Time to keep the role assignments for users authenticating via this identity provider. When not provided, global default configured in the Identity service will be used. Available since Identity API version 3.14 (Ussuri).'))
        enable_identity_provider = parser.add_mutually_exclusive_group()
        enable_identity_provider.add_argument('--enable', dest='enabled', action='store_true', default=True, help=_('Enable identity provider (default)'))
        enable_identity_provider.add_argument('--disable', dest='enabled', action='store_false', help=_('Disable the identity provider'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.remote_id_file:
            file_content = utils.read_blob_file_contents(parsed_args.remote_id_file)
            remote_ids = file_content.splitlines()
            remote_ids = list(map(str.strip, remote_ids))
        else:
            remote_ids = parsed_args.remote_id if parsed_args.remote_id else None
        domain_id = None
        if parsed_args.domain:
            domain_id = common.find_domain(identity_client, parsed_args.domain).id
        kwargs = {}
        auth_ttl = parsed_args.authorization_ttl
        if auth_ttl is not None:
            if auth_ttl < 0:
                msg = _('%(param)s must be positive integer or zero.') % {'param': 'authorization-ttl'}
                raise exceptions.CommandError(msg)
            kwargs['authorization_ttl'] = auth_ttl
        idp = identity_client.federation.identity_providers.create(id=parsed_args.identity_provider_id, remote_ids=remote_ids, description=parsed_args.description, domain_id=domain_id, enabled=parsed_args.enabled, **kwargs)
        idp._info.pop('links', None)
        remote_ids = format_columns.ListColumn(idp._info.pop('remote_ids', []))
        idp._info['remote_ids'] = remote_ids
        return zip(*sorted(idp._info.items()))