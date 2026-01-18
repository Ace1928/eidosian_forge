import logging
from blazarclient import command
from blazarclient import utils
class ShowAllocations(command.ShowCommand):
    """Show allocations for resource identified by type and ID."""
    resource = 'allocation'
    json_indent = 4
    id_pattern = RESOURCE_ID_PATTERN
    name_key = 'hypervisor_hostname'
    log = logging.getLogger(__name__ + '.ShowHostAllocation')

    def get_parser(self, prog_name):
        parser = super(ShowAllocations, self).get_parser(prog_name)
        parser.add_argument('resource_type', choices=['host'], help='Show allocations for a resource type')
        if self.allow_names:
            help_str = 'ID or name of %s to look up'
        else:
            help_str = 'ID of %s to look up'
        parser.add_argument('id', metavar='RESOURCE', help=help_str % 'resource')
        parser.add_argument('--reservation-id', dest='reservation_id', default=None, help='Show only allocations with specific reservation_id')
        parser.add_argument('--lease-id', dest='lease_id', default=None, help='Show only allocations with specific lease_id')
        return parser

    def get_data(self, parsed_args):
        self.log.debug('get_data(%s)' % parsed_args)
        blazar_client = self.get_client()
        resource_manager = getattr(blazar_client, self.resource)
        if self.allow_names:
            res_id = utils.find_resource_id_by_name_or_id(blazar_client, parsed_args.resource_type, parsed_args.id, self.name_key, self.id_pattern)
        else:
            res_id = parsed_args.id
        data = resource_manager.get(self.args2body(parsed_args)['resource'], res_id)
        if parsed_args.lease_id is not None:
            data['reservations'] = list(filter(lambda d: d['lease_id'] == parsed_args.lease_id, data['reservations']))
        if parsed_args.reservation_id is not None:
            data['reservations'] = list(filter(lambda d: d['id'] == parsed_args.reservation_id, data['reservations']))
        self.format_output_data(data)
        return list(zip(*sorted(data.items())))

    def args2body(self, parsed_args):
        params = {}
        if parsed_args.resource_type == 'host':
            params.update(dict(resource='os-hosts'))
        return params