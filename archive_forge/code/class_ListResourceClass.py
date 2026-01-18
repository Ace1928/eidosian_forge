from osc_lib.command import command
from osc_lib import utils
from osc_placement import version
class ListResourceClass(command.Lister):
    """Return a list of all resource classes.

    This command requires at least ``--os-placement-api-version 1.2``.
    """

    def get_parser(self, prog_name):
        parser = super(ListResourceClass, self).get_parser(prog_name)
        return parser

    @version.check(version.ge('1.2'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        resource_classes = http.request('GET', BASE_URL).json()['resource_classes']
        rows = (utils.get_dict_properties(i, FIELDS) for i in resource_classes)
        return (FIELDS, rows)