from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import text
from six.moves import zip
def _GetCommonScopeNameForRefs(self, refs):
    """Gets common scope for references."""
    has_zone = any((hasattr(ref, 'zone') for ref in refs))
    has_region = any((hasattr(ref, 'region') for ref in refs))
    if has_zone and (not has_region):
        return 'zone'
    elif has_region and (not has_zone):
        return 'region'
    else:
        return None