from osc_lib.command import command
from osc_lib import utils
from osc_placement import version
class SetResourceClass(command.Command):
    """Create or validate the existence of single resource class.

    Unlike ``openstack resource class create``, this command also succeeds if
    the resource class already exists, which makes this an idempotent check or
    create command.

    This command requires at least ``--os-placement-api-version 1.7``.
    """

    def get_parser(self, prog_name):
        parser = super(SetResourceClass, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help='Name of the resource class')
        return parser

    @version.check(version.ge('1.7'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL + '/' + parsed_args.name
        http.request('PUT', url)