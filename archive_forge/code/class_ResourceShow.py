from cliff import lister
from cliff import show
from vitrageclient.common import utils
class ResourceShow(show.ShowOne):
    """Show a resource"""

    def get_parser(self, prog_name):
        parser = super(ResourceShow, self).get_parser(prog_name)
        parser.add_argument('vitrage_id', help='vitrage_id of a resource')
        return parser

    def take_action(self, parsed_args):
        vitrage_id = parsed_args.vitrage_id
        resource = utils.get_client(self).resource.get(vitrage_id=vitrage_id)
        return self.dict2columns(resource)