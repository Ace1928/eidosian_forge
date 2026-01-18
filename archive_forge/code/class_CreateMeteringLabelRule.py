from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class CreateMeteringLabelRule(neutronv20.CreateCommand):
    """Create a metering label rule for a given label."""
    resource = 'metering_label_rule'

    def add_known_arguments(self, parser):
        parser.add_argument('label_id', metavar='LABEL', help=_('ID or name of the label.'))
        parser.add_argument('remote_ip_prefix', metavar='REMOTE_IP_PREFIX', help=_('CIDR to match on.'))
        parser.add_argument('--direction', default='ingress', choices=['ingress', 'egress'], type=utils.convert_to_lowercase, help=_('Direction of traffic, default: ingress.'))
        parser.add_argument('--excluded', action='store_true', help=_('Exclude this CIDR from the label, default: not excluded.'))

    def args2body(self, parsed_args):
        neutron_client = self.get_client()
        label_id = neutronv20.find_resourceid_by_name_or_id(neutron_client, 'metering_label', parsed_args.label_id)
        body = {'metering_label_id': label_id, 'remote_ip_prefix': parsed_args.remote_ip_prefix}
        neutronv20.update_dict(parsed_args, body, ['direction', 'excluded'])
        return {'metering_label_rule': body}