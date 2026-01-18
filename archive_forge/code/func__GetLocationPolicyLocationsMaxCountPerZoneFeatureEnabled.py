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
def _GetLocationPolicyLocationsMaxCountPerZoneFeatureEnabled(args, messages):
    """Helper function for getting location for location policy."""
    locations = []
    if args.location_policy:
        for zone, policy in args.location_policy.items():
            zone_policy = arg_utils.ChoiceToEnum(policy, messages.LocationPolicyLocation.PreferenceValueValuesEnum)
            if args.max_count_per_zone and zone in args.max_count_per_zone:
                locations.append(messages.LocationPolicy.LocationsValue.AdditionalProperty(key='zones/{}'.format(zone), value=messages.LocationPolicyLocation(preference=zone_policy, constraints=messages.LocationPolicyLocationConstraints(maxCount=int(args.max_count_per_zone[zone])))))
            else:
                locations.append(messages.LocationPolicy.LocationsValue.AdditionalProperty(key='zones/{}'.format(zone), value=messages.LocationPolicyLocation(preference=zone_policy)))
    zone_policy_allowed_preference = arg_utils.ChoiceToEnum('allow', messages.LocationPolicyLocation.PreferenceValueValuesEnum)
    if args.max_count_per_zone:
        for zone, count in args.max_count_per_zone.items():
            if args.location_policy and zone not in args.location_policy or not args.location_policy:
                locations.append(messages.LocationPolicy.LocationsValue.AdditionalProperty(key='zones/{}'.format(zone), value=messages.LocationPolicyLocation(preference=zone_policy_allowed_preference, constraints=messages.LocationPolicyLocationConstraints(maxCount=int(count)))))
    return messages.LocationPolicy.LocationsValue(additionalProperties=locations)