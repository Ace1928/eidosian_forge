import os
from cliff import command
from cliff import lister
from cliff import show
from barbicanclient.v1 import secrets
class ListSecret(lister.Lister):
    """List secrets."""

    def get_parser(self, prog_name):
        parser = super(ListSecret, self).get_parser(prog_name)
        parser.add_argument('--limit', '-l', default=10, help='specify the limit to the number of items to list per page (default: %(default)s; maximum: 100)', type=int)
        parser.add_argument('--offset', '-o', default=0, help='specify the page offset (default: %(default)s)', type=int)
        parser.add_argument('--name', '-n', default=None, help='specify the secret name (default: %(default)s)')
        parser.add_argument('--algorithm', '-a', default=None, help='the algorithm filter for the list(default: %(default)s).')
        parser.add_argument('--bit-length', '-b', default=0, help='the bit length filter for the list (default: %(default)s).', type=int)
        parser.add_argument('--mode', '-m', default=None, help='the algorithm mode filter for the list (default: %(default)s).')
        parser.add_argument('--secret-type', '-s', default=None, help='specify the secret type (default: %(default)s).')
        return parser

    def take_action(self, args):
        obj_list = self.app.client_manager.key_manager.secrets.list(limit=args.limit, offset=args.offset, name=args.name, algorithm=args.algorithm, mode=args.mode, bits=args.bit_length, secret_type=args.secret_type)
        return secrets.Secret._list_objects(obj_list)