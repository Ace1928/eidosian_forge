from osc_lib.command import command
from osc_placement import version
class ShowTrait(command.ShowOne):
    """Check if a trait name exists in this cloud.

    This command requires at least ``--os-placement-api-version 1.6``.
    """

    def get_parser(self, prog_name):
        parser = super(ShowTrait, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help='Name of the trait.')
        return parser

    @version.check(version.ge('1.6'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = '/'.join([BASE_URL, parsed_args.name])
        http.request('GET', url)
        return (FIELDS, [parsed_args.name])