from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def ConstructNodeParameterConfigMessage(map_class, config_class, nodes_configs):
    """Constructs a node configs API message.

  Args:
    map_class: The map message class.
    config_class: The config (map-entry) message class.
    nodes_configs: The list of nodes configurations.

  Returns:
    The constructed message.
  """
    properties = []
    for nodes_config in nodes_configs:
        if nodes_config.count == 0:
            continue
        node_type_config = config_class(nodeCount=nodes_config.count)
        if nodes_config.custom_core_count > 0:
            node_type_config.customCoreCount = nodes_config.custom_core_count
        prop = map_class.AdditionalProperty(key=nodes_config.type, value=node_type_config)
        properties.append(prop)
    return map_class(additionalProperties=properties)