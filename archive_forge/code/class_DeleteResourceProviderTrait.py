from osc_lib.command import command
from osc_placement import version
class DeleteResourceProviderTrait(command.Command):
    """Dissociate all the traits from the resource provider.

    Note that this command is not atomic if multiple processes are managing
    traits for the same provider.

    This command requires at least ``--os-placement-api-version 1.6``.
    """

    def get_parser(self, prog_name):
        parser = super(DeleteResourceProviderTrait, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider.')
        return parser

    @version.check(version.ge('1.6'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = RP_TRAITS_URL.format(uuid=parsed_args.uuid)
        http.request('DELETE', url)