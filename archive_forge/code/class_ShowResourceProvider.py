import argparse
from osc_lib.command import command
from osc_lib import utils
from osc_placement.resources import common
from osc_placement import version
class ShowResourceProvider(command.ShowOne, version.CheckerMixin):
    """Show resource provider details"""

    def get_parser(self, prog_name):
        parser = super(ShowResourceProvider, self).get_parser(prog_name)
        parser.add_argument('--allocations', action='store_true', help='include the info on allocations of the provider resources')
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL + '/' + parsed_args.uuid
        resource = http.request('GET', url).json()
        fields = ('uuid', 'name', 'generation')
        if self.compare_version(version.ge('1.14')):
            fields += ('root_provider_uuid', 'parent_provider_uuid')
        if parsed_args.allocations:
            allocs_url = ALLOCATIONS_URL.format(uuid=parsed_args.uuid)
            allocs = http.request('GET', allocs_url).json()['allocations']
            resource['allocations'] = allocs
            fields += ('allocations',)
        return (fields, utils.get_dict_properties(resource, fields))