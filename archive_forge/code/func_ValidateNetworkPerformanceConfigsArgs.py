from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def ValidateNetworkPerformanceConfigsArgs(args):
    """Validates advanced networking bandwidth tier values."""
    for config in getattr(args, 'network_performance_configs', []) or []:
        total_tier = config.get('total-egress-bandwidth-tier', '').upper()
        if total_tier and total_tier not in constants.ADV_NETWORK_TIER_CHOICES:
            raise exceptions.InvalidArgumentException('--network-performance-configs', 'Invalid total-egress-bandwidth-tier tier value, "{tier}".\n             Tier value must be one of the following {tier_values}'.format(tier=total_tier, tier_values=','.join([six.text_type(tier_val) for tier_val in constants.ADV_NETWORK_TIER_CHOICES])))