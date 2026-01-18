import os
from cliff import command
from cliff import lister
from cliff import show
from barbicanclient.v1 import secrets
class StoreSecret(show.ShowOne):
    """Store a secret in Barbican."""

    def get_parser(self, prog_name):
        parser = super(StoreSecret, self).get_parser(prog_name)
        parser.add_argument('--name', '-n', help='a human-friendly name.')
        parser.add_argument('--secret-type', '-s', default='opaque', help='the secret type; must be one of symmetric, public, private, certificate, passphrase, opaque (default)')
        parser.add_argument('--payload-content-type', '-t', help='the type/format of the provided secret data; "text/plain" is assumed to be UTF-8; required when --payload is supplied.')
        parser.add_argument('--payload-content-encoding', '-e', help='required if --payload-content-type is "application/octet-stream".')
        parser.add_argument('--algorithm', '-a', default='aes', help='the algorithm (default: %(default)s).')
        parser.add_argument('--bit-length', '-b', default=256, help='the bit length (default: %(default)s).', type=int)
        parser.add_argument('--mode', '-m', default='cbc', help='the algorithm mode; used only for reference (default: %(default)s)')
        parser.add_argument('--expiration', '-x', help='the expiration time for the secret in ISO 8601 format.')
        payload_params = parser.add_mutually_exclusive_group(required=False)
        payload_params.add_argument('--payload', '-p', help='the unencrypted secret data.')
        payload_params.add_argument('--file', '-F', metavar='<filename>', help='file containing the secret payload')
        return parser

    def take_action(self, args):
        data = None
        if args.file:
            with open(args.file, 'rb') as f:
                data = f.read()
        payload = args.payload.encode('utf-8') if args.payload else data
        entity = self.app.client_manager.key_manager.secrets.create(name=args.name, payload=payload, payload_content_type=args.payload_content_type, payload_content_encoding=args.payload_content_encoding, algorithm=args.algorithm, bit_length=args.bit_length, mode=args.mode, expiration=args.expiration, secret_type=args.secret_type)
        entity.store()
        return entity._get_formatted_entity()