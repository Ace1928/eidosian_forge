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
def AddSetTargetArgs(parser, include_psc_google_apis=False, include_target_service_attachment=False, include_regional_tcp_proxy=False):
    """Adds flags for the set-target command."""
    AddUpdateTargetArgs(parser, include_psc_google_apis, include_target_service_attachment, include_regional_tcp_proxy)

    def CreateDeprecationAction(name):
        return actions.DeprecationAction(name, warn="The {flag_name} option is deprecated and will be removed in an upcoming release. If you're currently using this argument, you should remove it from your workflows.", removed=False, action='store')
    parser.add_argument('--network', required=False, help='Only for --load-balancing-scheme=INTERNAL or --load-balancing-scheme=INTERNAL_SELF_MANAGED or --load-balancing-scheme=EXTERNAL_MANAGED (regional) or --load-balancing-scheme=INTERNAL_MANAGED) Network that this forwarding rule applies to. If this field is not specified, the default network is used. In the absence of the default network, this field must be specified.', action=CreateDeprecationAction('--network'))
    parser.add_argument('--subnet', required=False, help='Only for --load-balancing-scheme=INTERNAL and --load-balancing-scheme=INTERNAL_MANAGED) Subnetwork that this forwarding rule applies to. If the network is auto mode, this flag is optional. If the network is custom mode, this flag is required.', action=CreateDeprecationAction('--subnet'))
    parser.add_argument('--subnet-region', required=False, help='Region of the subnetwork to operate on. If not specified, the region is set to the region of the forwarding rule. Overrides the default compute/region property value for this command invocation.', action=CreateDeprecationAction('--subnet-region'))
    AddLoadBalancingScheme(parser, include_psc_google_apis=include_psc_google_apis, include_target_service_attachment=include_target_service_attachment, include_regional_tcp_proxy=include_regional_tcp_proxy, deprecation_action=CreateDeprecationAction('--load-balancing-scheme'))