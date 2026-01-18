from cliff import command
from cliff import lister
from cliff import show
class ListOrder(lister.Lister):
    """List orders."""

    def get_parser(self, prog_name):
        parser = super(ListOrder, self).get_parser(prog_name)
        parser.add_argument('--limit', '-l', default=10, help='specify the limit to the number of items to list per page (default: %(default)s; maximum: 100)', type=int)
        parser.add_argument('--offset', '-o', default=0, help='specify the page offset (default: %(default)s)', type=int)
        return parser

    def take_action(self, args):
        obj_list = self.app.client_manager.key_manager.orders.list(args.limit, args.offset)
        if not obj_list:
            return ([], [])
        columns = obj_list[0]._get_generic_columns()
        data = (obj._get_generic_data() for obj in obj_list)
        return (columns, data)