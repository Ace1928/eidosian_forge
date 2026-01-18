from osc_lib.command import command
from osc_placement import version
class ListResourceProviderTrait(command.Lister):
    """List traits associated with the resource provider identified by {uuid}.

    This command requires at least ``--os-placement-api-version 1.6``.
    """

    def get_parser(self, prog_name):
        parser = super(ListResourceProviderTrait, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider.')
        return parser

    @version.check(version.ge('1.6'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = RP_TRAITS_URL.format(uuid=parsed_args.uuid)
        traits = http.request('GET', url).json()['traits']
        return (FIELDS, [[t] for t in traits])