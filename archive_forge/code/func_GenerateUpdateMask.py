from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.batch import resource_allowances
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.batch import resource_args
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def GenerateUpdateMask(self, args):
    """Create Update Mask for ResourceAllowances."""
    update_mask = []
    if args.IsSpecified('usage_limit'):
        update_mask.append('usageResourceAllowance.spec.limit.limit')
    return update_mask