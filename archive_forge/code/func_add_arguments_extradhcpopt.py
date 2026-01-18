import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def add_arguments_extradhcpopt(self, parser):
    group_sg = parser.add_mutually_exclusive_group()
    group_sg.add_argument('--extra-dhcp-opt', default=[], action='append', dest='extra_dhcp_opts', type=utils.str2dict_type(required_keys=['opt_name'], optional_keys=['opt_value', 'ip_version']), help=_('Extra dhcp options to be assigned to this port: opt_name=<dhcp_option_name>,opt_value=<value>,ip_version={4,6}. You can repeat this option.'))