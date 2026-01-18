import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class UpdateExtraDhcpOptMixin(object):

    def add_arguments_extradhcpopt(self, parser):
        group_sg = parser.add_mutually_exclusive_group()
        group_sg.add_argument('--extra-dhcp-opt', default=[], action='append', dest='extra_dhcp_opts', type=utils.str2dict_type(required_keys=['opt_name'], optional_keys=['opt_value', 'ip_version']), help=_('Extra dhcp options to be assigned to this port: opt_name=<dhcp_option_name>,opt_value=<value>,ip_version={4,6}. You can repeat this option.'))

    def args2body_extradhcpopt(self, parsed_args, port):
        ops = []
        if parsed_args.extra_dhcp_opts:
            opt_ele = {}
            edo_err_msg = _('Invalid --extra-dhcp-opt option, can only be: opt_name=<dhcp_option_name>,opt_value=<value>,ip_version={4,6}. You can repeat this option.')
            for opt in parsed_args.extra_dhcp_opts:
                opt_ele.update(opt)
                if 'opt_name' in opt_ele and ('opt_value' in opt_ele or 'ip_version' in opt_ele):
                    if opt_ele.get('opt_value') == 'null':
                        opt_ele['opt_value'] = None
                    ops.append(opt_ele)
                    opt_ele = {}
                else:
                    raise exceptions.CommandError(edo_err_msg)
        if ops:
            port['extra_dhcp_opts'] = ops