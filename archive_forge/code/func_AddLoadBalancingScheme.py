from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def AddLoadBalancingScheme(parser, include_psc_google_apis=False, include_target_service_attachment=False, include_regional_tcp_proxy=False, deprecation_action=None):
    """Adds the load-balancing-scheme flag."""
    td_proxies = '--target-http-proxy, --target-https-proxy, --target-grpc-proxy, --target-tcp-proxy'
    ilb_proxies = '--target-http-proxy, --target-https-proxy'
    if include_regional_tcp_proxy:
        ilb_proxies += ', --target-tcp-proxy'
    load_balancing_choices = {'EXTERNAL': 'Classic Application Load Balancers, global external proxy Network  Load Balancers, external passthrough Network Load Balancers or  protocol forwarding, used with one of --target-http-proxy, --target-https-proxy, --target-tcp-proxy, --target-ssl-proxy, --target-pool, --target-vpn-gateway, --target-instance.', 'EXTERNAL_MANAGED': 'Global and regional external Application Load Balancers, and regional external proxy Network Load Balancers, used with --target-http-proxy, --target-https-proxy, --target-tcp-proxy.', 'INTERNAL': 'Internal passthrough Network Load Balancers or protocol forwarding, used with --backend-service.', 'INTERNAL_SELF_MANAGED': 'Traffic Director, used with {0}.'.format(td_proxies), 'INTERNAL_MANAGED': 'Internal Application Load Balancers and internal proxy Network Load Balancers, used with {0}.'.format(ilb_proxies)}
    include_psc = include_psc_google_apis or include_target_service_attachment
    help_text_with_psc = "This defines the forwarding rule's load balancing scheme. Note that it defaults to EXTERNAL and is not applicable for Private Service Connect forwarding rules."
    help_text_disabled_psc = "This defines the forwarding rule's load balancing scheme."
    parser.add_argument('--load-balancing-scheme', choices=load_balancing_choices, type=lambda x: x.replace('-', '_').upper(), default=None if include_psc else 'EXTERNAL', help=help_text_with_psc if include_psc else help_text_disabled_psc, action=deprecation_action)