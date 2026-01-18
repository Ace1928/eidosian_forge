from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.command_lib.compute.routers import router_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _UpdateBgpPeerBfdMessage(messages, peer, args):
    """Updates BGP peer BFD messages based on flag arguments."""
    if not (args.IsSpecified('bfd_min_receive_interval') or args.IsSpecified('bfd_min_transmit_interval') or args.IsSpecified('bfd_session_initialization_mode') or args.IsSpecified('bfd_multiplier')):
        return None
    if peer.bfd is not None:
        bfd = peer.bfd
    else:
        bfd = messages.RouterBgpPeerBfd()
    attrs = {}
    if args.bfd_session_initialization_mode is not None:
        attrs['sessionInitializationMode'] = messages.RouterBgpPeerBfd.SessionInitializationModeValueValuesEnum(args.bfd_session_initialization_mode)
    attrs['minReceiveInterval'] = args.bfd_min_receive_interval
    attrs['minTransmitInterval'] = args.bfd_min_transmit_interval
    attrs['multiplier'] = args.bfd_multiplier
    for attr, value in attrs.items():
        if value is not None:
            setattr(bfd, attr, value)
    return bfd