from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.immersive_stream.xr import api_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GenerateTargetLocationConfigs(release_track, add_region_configs, update_region_configs, remove_regions, current_instance):
    """Generates the target location configs.

  Args:
    release_track: ALPHA or GA release track
    add_region_configs: List of region config dicts of the form: [{'region':
      region1, 'capacity': capacity1, 'enable_autoscaling': enable_autoscaling1,
      'autoscaling_buffer': autoscaling_buffer1, 'autoscaling_min_capacity':
      autoscaling_min_capacity1}] that specifies the regions to add to the
      service instance
    update_region_configs: List of region config dicts of the form: [{'region':
      region1, 'capacity': capacity1}] that specifies the regions to update to
      the service instance
    remove_regions: List of regions to remove
    current_instance: instance object - current state of the service instance
      before update

  Returns:
    A LocationConfigsValue, with entries sorted by location
  """
    if current_instance is not None:
        additonal_properties = current_instance.locationConfigs.additionalProperties
        location_configs = {location_config.key: location_config.value for location_config in additonal_properties}
    else:
        location_configs = {}
    if add_region_configs:
        if any((region_config['region'] in location_configs for region_config in add_region_configs)):
            log.status.Print('Only new regions can be added.')
            return
        region_configs_diff = add_region_configs
    elif remove_regions:
        if any((region not in location_configs for region in remove_regions)):
            log.status.Print('Only existing regions can be removed.')
            return None
        region_configs_diff = ({'region': region, 'capacity': 0, 'enable_autoscaling': False} for region in remove_regions)
    elif update_region_configs:
        if any((region_config['region'] not in location_configs for region_config in update_region_configs)):
            log.status.Print('Only existing regions can be updated.')
            return None
        region_configs_diff = update_region_configs
    messages = api_util.GetMessages(release_track)
    location_configs_diff = messages.StreamInstance.LocationConfigsValue()
    for region_config in region_configs_diff:
        region = region_config['region']
        capacity = int(region_config['capacity'])
        enable_autoscaling = region_config.get('enable_autoscaling', False)
        location_config = messages.LocationConfig(location=region, capacity=capacity, enableAutoscaling=enable_autoscaling)
        if enable_autoscaling:
            location_config.autoscalingBuffer = int(region_config['autoscaling_buffer'])
            location_config.autoscalingMinCapacity = int(region_config['autoscaling_min_capacity'])
        location_configs_diff.additionalProperties.append(messages.StreamInstance.LocationConfigsValue.AdditionalProperty(key=region, value=location_config))
    for location_config in location_configs_diff.additionalProperties:
        if remove_regions and location_config.value.capacity == 0:
            location_configs.pop(location_config.key, None)
        else:
            location_configs[location_config.key] = location_config.value
    target_location_configs = messages.StreamInstance.LocationConfigsValue()
    for key, location_config in sorted(location_configs.items()):
        target_location_configs.additionalProperties.append(messages.StreamInstance.LocationConfigsValue.AdditionalProperty(key=key, value=location_config))
    return target_location_configs