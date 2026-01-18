from cliff import lister
from cliff import show
from barbicanclient.v1 import cas
class GetCA(show.ShowOne):
    """Retrieve a CA by providing its URI."""

    def get_parser(self, prog_name):
        parser = super(GetCA, self).get_parser(prog_name)
        parser.add_argument('URI', help='The URI reference for the CA.')
        return parser

    def take_action(self, args):
        entity = self.app.client_manager.key_manager.cas.get(ca_ref=args.URI)
        return entity._get_formatted_entity()