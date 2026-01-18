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
def AddUpdateTargetArgs(parser, include_psc_google_apis=False, include_target_service_attachment=False, include_regional_tcp_proxy=False):
    """Adds common flags for mutating forwarding rule targets."""
    target = parser.add_mutually_exclusive_group(required=True)
    TargetGrpcProxyArg().AddArgument(parser, mutex_group=target)
    if include_target_service_attachment:
        TargetServiceAttachmentArg().AddArgument(parser, mutex_group=target)
    TargetHttpProxyArg().AddArgument(parser, mutex_group=target)
    TargetHttpsProxyArg().AddArgument(parser, mutex_group=target)
    TARGET_INSTANCE_ARG.AddArgument(parser, mutex_group=target)
    TARGET_POOL_ARG.AddArgument(parser, mutex_group=target)
    TARGET_SSL_PROXY_ARG.AddArgument(parser, mutex_group=target)
    TargetTcpProxyArg(allow_regional=include_regional_tcp_proxy).AddArgument(parser, mutex_group=target)
    TARGET_VPN_GATEWAY_ARG.AddArgument(parser, mutex_group=target)
    BACKEND_SERVICE_ARG.AddArgument(parser, mutex_group=target)
    if include_psc_google_apis:
        target.add_argument('--target-google-apis-bundle', required=False, help='Target bundle of Google APIs that will receive forwarded traffic via Private Service Connect. Acceptable values are all-apis, meaning all Google APIs, or vpc-sc, meaning just the APIs that support VPC Service Controls', action='store')