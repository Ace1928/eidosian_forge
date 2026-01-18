from osc_lib.command import command
from osc_lib import exceptions
from osc_placement import version
class ListAggregate(command.Lister):
    """List resource provider aggregates.

    This command requires at least ``--os-placement-api-version 1.1``.
    """

    def get_parser(self, prog_name):
        parser = super(ListAggregate, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        return parser

    @version.check(version.ge('1.1'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL.format(uuid=parsed_args.uuid)
        resp = http.request('GET', url).json()
        return (FIELDS, [[r] for r in resp['aggregates']])