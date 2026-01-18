from osc_lib.command import command
from osc_lib import utils
from osc_placement import version
class ResourceShowUsage(command.Lister, version.CheckerMixin):
    """Show resource usages for a project (and optionally user) per class.

    Gives a report of usage information for resources associated with the
    project identified by the ``project_id`` argument and user identified by
    the ``--user-id`` option.

    This command requires at least ``--os-placement-api-version 1.9``.

    """

    def get_parser(self, prog_name):
        parser = super(ResourceShowUsage, self).get_parser(prog_name)
        parser.add_argument('project_id', metavar='<project-uuid>', help='ID of the project.')
        parser.add_argument('--user-id', metavar='<user-uuid>', help='ID of the user.')
        return parser

    @version.check(version.ge('1.9'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = USAGES_URL
        params = {'project_id': parsed_args.project_id}
        if parsed_args.user_id:
            params['user_id'] = parsed_args.user_id
        per_class = http.request('GET', url, params=params).json()['usages']
        usages = [{'resource_class': k, 'usage': v} for k, v in per_class.items()]
        rows = (utils.get_dict_properties(u, FIELDS) for u in usages)
        return (FIELDS, rows)