from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six.moves.urllib.parse
def ValidateRegionConfigArgs(region_configs, operation_name):
    """Validates the region config args do not contain duplicate regions and have valid autoscaling configuration, if enabled.

  Args:
    region_configs: Either add_region or update_region ArgList from the
      instance update args
    operation_name: String indicating if operation is an add or update region
      operation

  Returns:
    True if the region_configs are valid. False if not.
  """
    regions = {}
    for region_config in region_configs:
        regions[region_config['region']] = region_config
        if region_config.get('enable_autoscaling', False) and (not ('autoscaling_buffer' in region_config and 'autoscaling_min_capacity' in region_config)):
            log.error('Must set autoscaling_buffer and autoscaling_min_capacity if enable_autoscaling is set to true.')
            return False
    if len(regions) < len(region_configs):
        log.error('Duplicate regions in --{}-region arguments.'.format(operation_name))
        return False
    return True