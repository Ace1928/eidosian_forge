from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.command_lib.compute.routers import router_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def _CreateBgpPeerMessage(messages, args, md5_authentication_key_name, support_bfd_mode=False, instance_ref=None, enable_ipv6_bgp=False, enable_route_policies=False):
    """Creates a BGP peer with base attributes based on flag arguments.

  Args:
    messages: API messages holder.
    args: contains arguments passed to the command.
    md5_authentication_key_name: The md5 authentication key name.
    support_bfd_mode: The flag to indicate whether bfd mode is supported.
    instance_ref: An instance reference.
    enable_ipv6_bgp: The flag to indicate whether IPv6-based BGP is supported.
    enable_route_policies: The flag to indicate whether exportPolicies and
      importPolicies are supported.

  Returns:
    the RouterBgpPeer
  """
    if support_bfd_mode:
        bfd = _CreateBgpPeerBfdMessageMode(messages, args)
    else:
        bfd = _CreateBgpPeerBfdMessage(messages, args)
    enable = None
    if args.enabled is not None:
        if args.enabled:
            enable = messages.RouterBgpPeer.EnableValueValuesEnum.TRUE
        else:
            enable = messages.RouterBgpPeer.EnableValueValuesEnum.FALSE
    result = messages.RouterBgpPeer(name=args.peer_name, interfaceName=args.interface, peerIpAddress=args.peer_ip_address, peerAsn=args.peer_asn, advertisedRoutePriority=args.advertised_route_priority, enable=enable, bfd=bfd, enableIpv6=args.enable_ipv6, ipv6NexthopAddress=args.ipv6_nexthop_address, peerIpv6NexthopAddress=args.peer_ipv6_nexthop_address)
    if enable_ipv6_bgp:
        result.enableIpv4 = args.enable_ipv4
        result.ipv4NexthopAddress = args.ipv4_nexthop_address
        result.peerIpv4NexthopAddress = args.peer_ipv4_nexthop_address
    result.customLearnedRoutePriority = args.custom_learned_route_priority
    if instance_ref is not None:
        result.routerApplianceInstance = instance_ref.SelfLink()
    if args.md5_authentication_key is not None:
        result.md5AuthenticationKeyName = md5_authentication_key_name
    if enable_route_policies:
        if args.export_policies is not None:
            result.exportPolicies = args.export_policies
        if args.import_policies is not None:
            result.importPolicies = args.import_policies
    return result