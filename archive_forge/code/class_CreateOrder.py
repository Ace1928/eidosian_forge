from cliff import command
from cliff import lister
from cliff import show
class CreateOrder(show.ShowOne):
    """Create a new order."""

    def get_parser(self, prog_name):
        parser = super(CreateOrder, self).get_parser(prog_name)
        parser.add_argument('type', help='the type of the order (key, asymmetric, certificate) to create.')
        parser.add_argument('--name', '-n', help='a human-friendly name.')
        parser.add_argument('--algorithm', '-a', default='aes', help='the algorithm to be used with the requested key (default: %(default)s).')
        parser.add_argument('--bit-length', '-b', default=256, help='the bit length of the requested secret key (default: %(default)s).', type=int)
        parser.add_argument('--mode', '-m', default='cbc', help='the algorithm mode to be used with the requested key (default: %(default)s).')
        parser.add_argument('--payload-content-type', '-t', default='application/octet-stream', help='the type/format of the secret to be generated (default: %(default)s).')
        parser.add_argument('--expiration', '-x', help='the expiration time for the secret in ISO 8601 format.')
        parser.add_argument('--request-type', help='the type of the certificate request.')
        parser.add_argument('--subject-dn', help='the subject of the certificate.')
        parser.add_argument('--source-container-ref', help='the source of the certificate when using stored-key requests.')
        parser.add_argument('--ca-id', help='the identifier of the CA to use for the certificate request.')
        parser.add_argument('--profile', help='the profile of certificate to use.')
        parser.add_argument('--request-file', help='the file containing the CSR.')
        return parser

    def take_action(self, args):
        if args.type == 'certificate':
            request_data = None
            if args.request_file:
                try:
                    request_data = open(args.request_file, 'r').read()
                except IOError:
                    raise ValueError("Couldn't read request file %s." % args.request_file)
            entity = self.app.client_manager.key_manager.orders.create(name=args.name, type=args.type, subject_dn=args.subject_dn, request_type=args.request_type, source_container_ref=args.source_container_ref, ca_id=args.ca_id, profile=args.profile, request_data=request_data)
        else:
            entity = self.app.client_manager.key_manager.orders.create(name=args.name, type=args.type, payload_content_type=args.payload_content_type, algorithm=args.algorithm, bit_length=args.bit_length, mode=args.mode, expiration=args.expiration)
        entity.submit()
        return entity._get_formatted_entity()