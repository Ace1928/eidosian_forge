from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class RetrieveLoadBalancerStatus(neutronV20.NeutronCommand):
    """Retrieve status for a given loadbalancer.

    The only output is a formatted JSON tree, and the table format
    does not support this type of data.
    """
    resource = 'loadbalancer'

    def get_parser(self, prog_name):
        parser = super(RetrieveLoadBalancerStatus, self).get_parser(prog_name)
        parser.add_argument(self.resource, metavar=self.resource.upper(), help=_('ID or name of %s to show.') % self.resource)
        return parser

    def take_action(self, parsed_args):
        self.log.debug('run(%s)', parsed_args)
        neutron_client = self.get_client()
        lb_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, self.resource, parsed_args.loadbalancer)
        params = {}
        data = neutron_client.retrieve_loadbalancer_status(lb_id, **params)
        res = data['statuses']
        if 'statuses' in data:
            print(jsonutils.dumps(res, indent=4))