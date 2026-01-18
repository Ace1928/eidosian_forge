import argparse
from osc_lib.command import command
from osc_lib import utils
from osc_placement.resources import common
from osc_placement import version
class ListResourceProvider(command.Lister, version.CheckerMixin):
    """List resource providers"""

    def get_parser(self, prog_name):
        parser = super(ListResourceProvider, self).get_parser(prog_name)
        parser.add_argument('--uuid', metavar='<uuid>', help='UUID of the resource provider')
        parser.add_argument('--name', metavar='<name>', help='Name of the resource provider')
        parser.add_argument('--resource', metavar='<resource_class>=<value>', default=[], action='append', help='A resource class value pair indicating an amount of resource of a specified class that a provider must have the capacity to serve. May be repeated.\n\nThis param requires at least ``--os-placement-api-version 1.4``.')
        parser.add_argument('--in-tree', metavar='<in_tree>', help='Restrict listing to the same "provider tree" as the specified provider UUID. This option requires at least ``--os-placement-api-version 1.14``.')
        parser.add_argument('--required', metavar='<required>', action='append', default=[], help='A required trait. May be repeated. Resource providers must collectively contain all of the required traits. This option requires at least ``--os-placement-api-version 1.18``. Since ``--os-placement-api-version 1.39`` the value of this parameter can be a comma separated list of trait names to express OR relationship between those traits.')
        parser.add_argument('--forbidden', metavar='<forbidden>', action='append', default=[], help='A forbidden trait. May be repeated. Returned resource providers must not contain any of the specified traits. This option requires at least ``--os-placement-api-version 1.22``.')
        aggregate_group = parser.add_mutually_exclusive_group()
        aggregate_group.add_argument('--member-of', default=[], action='append', metavar='<member_of>', help='A list of comma-separated UUIDs of the resource provider aggregates. The returned resource providers must be associated with at least one of the aggregates identified by uuid. This param requires at least ``--os-placement-api-version 1.3`` and can be repeated to add(restrict) the condition with ``--os-placement-api-version 1.24`` or greater. For example, to get candidates either in agg1 or in agg2 and definitely in agg3, specify:\n\n``--member_of <agg1>,<agg2> --member_of <agg3>``')
        aggregate_group.add_argument('--aggregate-uuid', default=[], action='append', metavar='<aggregate_uuid>', help=argparse.SUPPRESS)
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        filters = {}
        if parsed_args.name:
            filters['name'] = parsed_args.name
        if parsed_args.uuid:
            filters['uuid'] = parsed_args.uuid
        if parsed_args.aggregate_uuid:
            self.check_version(version.ge('1.3'))
            self.deprecated_option_warning('--aggregate-uuid', '--member-of')
            filters['member_of'] = 'in:' + ','.join(parsed_args.aggregate_uuid)
        if parsed_args.resource:
            self.check_version(version.ge('1.4'))
            filters['resources'] = ','.join((resource.replace('=', ':') for resource in parsed_args.resource))
        if 'in_tree' in parsed_args and parsed_args.in_tree:
            self.check_version(version.ge('1.14'))
            filters['in_tree'] = parsed_args.in_tree
        required_traits = []
        if 'required' in parsed_args and parsed_args.required:
            self.check_version(version.ge('1.18'))
            if any((',' in required for required in parsed_args.required)):
                self.check_version(version.ge('1.39'))
            required_traits = parsed_args.required
        forbidden_traits = []
        if 'forbidden' in parsed_args and parsed_args.forbidden:
            self.check_version(version.ge('1.22'))
            forbidden_traits = ['!' + f for f in parsed_args.forbidden]
        filters['required'] = common.get_required_query_param_from_args(required_traits, forbidden_traits)
        if 'member_of' in parsed_args and parsed_args.member_of:
            self.check_version(version.ge('1.3'))
            filters['member_of'] = ['in:' + aggs for aggs in parsed_args.member_of]
        resources = http.request('GET', BASE_URL, params=filters).json()['resource_providers']
        fields = ('uuid', 'name', 'generation')
        if self.compare_version(version.ge('1.14')):
            fields += ('root_provider_uuid', 'parent_provider_uuid')
        rows = (utils.get_dict_properties(r, fields) for r in resources)
        return (fields, rows)