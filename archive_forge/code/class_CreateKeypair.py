import collections
import io
import logging
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class CreateKeypair(command.ShowOne):
    _description = _('Create new public or private key for server ssh access')

    def get_parser(self, prog_name):
        parser = super(CreateKeypair, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('New public or private key name'))
        key_group = parser.add_mutually_exclusive_group()
        key_group.add_argument('--public-key', metavar='<file>', help=_('Filename for public key to add. If not used, generates a private key in ssh-ed25519 format. To generate keys in other formats, including the legacy ssh-rsa format, you must use an external tool such as ssh-keygen and specify this argument.'))
        key_group.add_argument('--private-key', metavar='<file>', help=_('Filename for private key to save. If not used, print private key in console.'))
        parser.add_argument('--type', metavar='<type>', choices=['ssh', 'x509'], help=_('Keypair type (supported by --os-compute-api-version 2.2 or above)'))
        parser.add_argument('--user', metavar='<user>', help=_('The owner of the keypair (admin only) (name or ID) (supported by --os-compute-api-version 2.10 or above)'))
        identity_common.add_user_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        identity_client = self.app.client_manager.identity
        kwargs = {'name': parsed_args.name}
        if parsed_args.public_key:
            generated_keypair = None
            try:
                with io.open(os.path.expanduser(parsed_args.public_key)) as p:
                    public_key = p.read()
            except IOError as e:
                msg = _('Key file %(public_key)s not found: %(exception)s')
                raise exceptions.CommandError(msg % {'public_key': parsed_args.public_key, 'exception': e})
            kwargs['public_key'] = public_key
        else:
            generated_keypair = _generate_keypair()
            kwargs['public_key'] = generated_keypair.public_key
            if parsed_args.private_key:
                try:
                    with io.open(os.path.expanduser(parsed_args.private_key), 'w+') as p:
                        p.write(generated_keypair.private_key)
                except IOError as e:
                    msg = _('Key file %(private_key)s can not be saved: %(exception)s')
                    raise exceptions.CommandError(msg % {'private_key': parsed_args.private_key, 'exception': e})
        if parsed_args.type:
            if not sdk_utils.supports_microversion(compute_client, '2.2'):
                msg = _('--os-compute-api-version 2.2 or greater is required to support the --type option')
                raise exceptions.CommandError(msg)
            kwargs['key_type'] = parsed_args.type
        if parsed_args.user:
            if not sdk_utils.supports_microversion(compute_client, '2.10'):
                msg = _('--os-compute-api-version 2.10 or greater is required to support the --user option')
                raise exceptions.CommandError(msg)
            kwargs['user_id'] = identity_common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        keypair = compute_client.create_keypair(**kwargs)
        if parsed_args.public_key or parsed_args.private_key:
            display_columns, columns = _get_keypair_columns(keypair, hide_pub_key=True, hide_priv_key=True)
            data = utils.get_item_properties(keypair, columns)
            return (display_columns, data)
        else:
            self.app.stdout.write(generated_keypair.private_key)
            return ({}, {})