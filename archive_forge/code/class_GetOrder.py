from cliff import command
from cliff import lister
from cliff import show
class GetOrder(show.ShowOne):
    """Retrieve an order by providing its URI."""

    def get_parser(self, prog_name):
        parser = super(GetOrder, self).get_parser(prog_name)
        parser.add_argument('URI', help='The URI reference order.')
        return parser

    def take_action(self, args):
        entity = self.app.client_manager.key_manager.orders.get(order_ref=args.URI)
        return entity._get_formatted_entity()