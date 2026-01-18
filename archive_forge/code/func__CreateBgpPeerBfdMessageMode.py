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
def _CreateBgpPeerBfdMessageMode(messages, args):
    """Creates a BGP peer with base attributes based on flag arguments."""
    if not (args.IsSpecified('bfd_min_receive_interval') or args.IsSpecified('bfd_min_transmit_interval') or args.IsSpecified('bfd_session_initialization_mode') or args.IsSpecified('bfd_multiplier')):
        return None
    mode = None
    bfd_session_initialization_mode = None
    if args.bfd_session_initialization_mode is not None:
        mode = messages.RouterBgpPeerBfd.ModeValueValuesEnum(args.bfd_session_initialization_mode)
        bfd_session_initialization_mode = messages.RouterBgpPeerBfd.SessionInitializationModeValueValuesEnum(args.bfd_session_initialization_mode)
    return messages.RouterBgpPeerBfd(minReceiveInterval=args.bfd_min_receive_interval, minTransmitInterval=args.bfd_min_transmit_interval, mode=mode, sessionInitializationMode=bfd_session_initialization_mode, multiplier=args.bfd_multiplier)