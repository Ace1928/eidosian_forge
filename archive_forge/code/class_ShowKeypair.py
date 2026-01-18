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
class ShowKeypair(command.ShowOne):
    _description = _('Display key details')

    def get_parser(self, prog_name):
        parser = super(ShowKeypair, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<key>', help=_('Public or private key to display (name only)'))
        parser.add_argument('--public-key', action='store_true', default=False, help=_('Show only bare public key paired with the generated key'))
        parser.add_argument('--user', metavar='<user>', help=_('The owner of the keypair. (admin only) (name or ID). Requires ``--os-compute-api-version`` 2.10 or greater.'))
        identity_common.add_user_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        identity_client = self.app.client_manager.identity
        kwargs = {}
        if parsed_args.user:
            if not sdk_utils.supports_microversion(compute_client, '2.10'):
                msg = _('--os-compute-api-version 2.10 or greater is required to support the --user option')
                raise exceptions.CommandError(msg)
            kwargs['user_id'] = identity_common.find_user(identity_client, parsed_args.user, parsed_args.user_domain).id
        keypair = compute_client.find_keypair(parsed_args.name, **kwargs, ignore_missing=False)
        if not parsed_args.public_key:
            display_columns, columns = _get_keypair_columns(keypair, hide_pub_key=True)
            data = utils.get_item_properties(keypair, columns)
            return (display_columns, data)
        else:
            self.app.stdout.write(keypair.public_key)
            return ({}, {})