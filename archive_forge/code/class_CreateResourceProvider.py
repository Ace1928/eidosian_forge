import argparse
from osc_lib.command import command
from osc_lib import utils
from osc_placement.resources import common
from osc_placement import version
class CreateResourceProvider(command.ShowOne, version.CheckerMixin):
    """Create a new resource provider"""

    def get_parser(self, prog_name):
        parser = super(CreateResourceProvider, self).get_parser(prog_name)
        parser.add_argument('--parent-provider', metavar='<parent_provider>', help='UUID of the parent provider. Omit for no parent. This option requires at least ``--os-placement-api-version 1.14``.')
        parser.add_argument('--uuid', metavar='<uuid>', help='UUID of the resource provider')
        parser.add_argument('name', metavar='<name>', help='Name of the resource provider')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        data = {'name': parsed_args.name}
        if 'uuid' in parsed_args and parsed_args.uuid:
            data['uuid'] = parsed_args.uuid
        if 'parent_provider' in parsed_args and parsed_args.parent_provider:
            self.check_version(version.ge('1.14'))
            data['parent_provider_uuid'] = parsed_args.parent_provider
        resp = http.request('POST', BASE_URL, json=data)
        resource = http.request('GET', resp.headers['Location']).json()
        fields = ('uuid', 'name', 'generation')
        if self.compare_version(version.ge('1.14')):
            fields += ('root_provider_uuid', 'parent_provider_uuid')
        return (fields, utils.get_dict_properties(resource, fields))