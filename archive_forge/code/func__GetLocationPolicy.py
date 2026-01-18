from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute import secure_tags_utils
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def _GetLocationPolicy(args, messages, target_shape_enabled, max_count_per_zone_enabled):
    """Get locationPolicy value as required by API."""
    if not (args.IsKnownAndSpecified('location_policy') or args.IsKnownAndSpecified('max_count_per_zone')) and (not target_shape_enabled or not args.IsKnownAndSpecified('target_distribution_shape')):
        return None
    location_policy = messages.LocationPolicy()
    if args.IsKnownAndSpecified('location_policy') or args.IsKnownAndSpecified('max_count_per_zone'):
        if max_count_per_zone_enabled:
            location_policy.locations = _GetLocationPolicyLocationsMaxCountPerZoneFeatureEnabled(args, messages)
        else:
            location_policy.locations = _GetLocationPolicyLocationsMaxCountPerZoneFeatureDisabled(args, messages)
    if target_shape_enabled and args.IsKnownAndSpecified('target_distribution_shape'):
        location_policy.targetShape = arg_utils.ChoiceToEnum(args.target_distribution_shape, messages.LocationPolicy.TargetShapeValueValuesEnum)
    return location_policy