from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_placement import version
class ShowAllocation(command.Lister, version.CheckerMixin):
    """Show resource allocations for a given consumer.

    Starting with ``--os-placement-api-version 1.12`` the API response contains
    the ``project_id`` and ``user_id`` of allocations which also appears in the
    CLI output.

    Starting with ``--os-placement-api-version 1.38`` the API response contains
    the ``consumer_type`` of consumer which also appears in the CLI output.
    """

    def get_parser(self, prog_name):
        parser = super(ShowAllocation, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the consumer')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL + '/' + parsed_args.uuid
        resp = http.request('GET', url).json()
        per_provider = resp['allocations'].items()
        props = {}
        fields = ('resource_provider', 'generation', 'resources')
        if self.compare_version(version.ge('1.12')):
            fields += ('project_id', 'user_id')
            props['project_id'] = resp.get('project_id')
            props['user_id'] = resp.get('user_id')
        if self.compare_version(version.ge('1.38')):
            fields += ('consumer_type',)
            props['consumer_type'] = resp.get('consumer_type')
        allocs = [dict(resource_provider=k, **props, **v) for k, v in per_provider]
        rows = (utils.get_dict_properties(a, fields) for a in allocs)
        return (fields, rows)