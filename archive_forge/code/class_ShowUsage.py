from osc_lib.command import command
from osc_lib import utils
from osc_placement import version
class ShowUsage(command.Lister):
    """Show resource usages per class for a given resource provider."""

    def get_parser(self, prog_name):
        parser = super(ShowUsage, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL.format(uuid=parsed_args.uuid)
        per_class = http.request('GET', url).json()['usages']
        usages = [{'resource_class': k, 'usage': v} for k, v in per_class.items()]
        rows = (utils.get_dict_properties(u, FIELDS) for u in usages)
        return (FIELDS, rows)