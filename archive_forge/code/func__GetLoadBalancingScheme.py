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
def _GetLoadBalancingScheme(args, messages, is_psc):
    """Get load balancing scheme."""
    if not args.load_balancing_scheme:
        return None if is_psc else messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.EXTERNAL
    if args.load_balancing_scheme == 'INTERNAL':
        return messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.INTERNAL
    elif args.load_balancing_scheme == 'EXTERNAL':
        return messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.EXTERNAL
    elif args.load_balancing_scheme == 'EXTERNAL_MANAGED':
        return messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.EXTERNAL_MANAGED
    elif args.load_balancing_scheme == 'INTERNAL_SELF_MANAGED':
        return messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.INTERNAL_SELF_MANAGED
    elif args.load_balancing_scheme == 'INTERNAL_MANAGED':
        return messages.ForwardingRule.LoadBalancingSchemeValueValuesEnum.INTERNAL_MANAGED
    return None