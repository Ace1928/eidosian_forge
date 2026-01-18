from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import forwarding_rules_utils as utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions as fw_exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.core import log
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _ConstructNetworkTier(messages, args):
    """Get network tier."""
    if args.network_tier:
        network_tier = args.network_tier.upper()
        if network_tier in constants.NETWORK_TIER_CHOICES_FOR_INSTANCE:
            return messages.ForwardingRule.NetworkTierValueValuesEnum(args.network_tier)
        else:
            raise exceptions.InvalidArgumentException('--network-tier', 'Invalid network tier [{tier}]'.format(tier=network_tier))
    else:
        return