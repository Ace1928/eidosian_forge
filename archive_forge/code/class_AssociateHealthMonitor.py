from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class AssociateHealthMonitor(neutronV20.NeutronCommand):
    """Create a mapping between a health monitor and a pool."""
    resource = 'health_monitor'

    def get_parser(self, prog_name):
        parser = super(AssociateHealthMonitor, self).get_parser(prog_name)
        parser.add_argument('health_monitor_id', metavar='HEALTH_MONITOR_ID', help=_('Health monitor to associate.'))
        parser.add_argument('pool_id', metavar='POOL', help=_('ID of the pool to be associated with the health monitor.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        body = {'health_monitor': {'id': parsed_args.health_monitor_id}}
        pool_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'pool', parsed_args.pool_id)
        neutron_client.associate_health_monitor(pool_id, body)
        print(_('Associated health monitor %s') % parsed_args.health_monitor_id, file=self.app.stdout)