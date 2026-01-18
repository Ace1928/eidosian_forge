from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_placement import version
class UnsetAllocation(command.Lister, version.CheckerMixin):
    """Removes one or more sets of provider allocations for a consumer.

    Note that omitting both the ``--provider`` and the ``--resource-class``
    option is equivalent to removing all allocations for the given consumer.

    This command requires ``--os-placement-api-version 1.12`` or greater. Use
    ``openstack resource provider allocation set`` for older versions.
    """

    def get_parser(self, prog_name):
        parser = super(UnsetAllocation, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<consumer_uuid>', help='UUID of the consumer. It is strongly recommended to use ``--os-placement-api-version 1.28`` or greater when using this option to ensure the other allocation information is retained. ')
        parser.add_argument('--provider', metavar='provider_uuid', action='append', default=[], help='UUID of a specific resource provider from which to remove allocations for the given consumer. This is useful when the consumer has allocations on more than one provider, for example after evacuating a server to another compute node and you want to cleanup allocations on the source compute node resource provider in order to delete it. Specify multiple times to remove allocations against multiple resource providers. Omit this option to remove all allocations for the consumer, or to remove all allocationsof a specific resource class from all the resource provider with the ``--resource_class`` option. ')
        parser.add_argument('--resource-class', metavar='resource_class', action='append', default=[], help='Name of a resource class from which to remove allocations for the given consumer. This is useful when the consumer has allocations on more than one resource class. By default, this will remove allocations for the given resource class from all the providers. If ``--provider`` option is also specified, allocations to remove will be limited to that resource class of the given resource provider.')
        return parser

    @version.check(version.ge('1.12'))
    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL + '/' + parsed_args.uuid
        payload = http.request('GET', url).json()
        allocations = payload['allocations']
        if parsed_args.resource_class:
            rp_uuids = set(allocations)
            if parsed_args.provider:
                rp_uuids &= set(parsed_args.provider)
            for rp_uuid in rp_uuids:
                for rc in parsed_args.resource_class:
                    allocations[rp_uuid]['resources'].pop(rc, None)
                if not allocations[rp_uuid]['resources']:
                    allocations.pop(rp_uuid, None)
        elif parsed_args.provider:
            for rp_uuid in parsed_args.provider:
                allocations.pop(rp_uuid, None)
        else:
            allocations = {}
        supports_consumer_generation = self.compare_version(version.ge('1.28'))
        if allocations or supports_consumer_generation:
            payload['allocations'] = allocations
            http.request('PUT', url, json=payload)
        else:
            http.request('DELETE', url)
        resp = http.request('GET', url).json()
        per_provider = resp['allocations'].items()
        props = {}
        fields = ('resource_provider', 'generation', 'resources', 'project_id', 'user_id')
        if self.compare_version(version.ge('1.38')):
            fields += ('consumer_type',)
            props['consumer_type'] = resp.get('consumer_type')
        allocs = [dict(project_id=resp['project_id'], user_id=resp['user_id'], resource_provider=k, **props, **v) for k, v in per_provider]
        rows = (utils.get_dict_properties(a, fields) for a in allocs)
        return (fields, rows)