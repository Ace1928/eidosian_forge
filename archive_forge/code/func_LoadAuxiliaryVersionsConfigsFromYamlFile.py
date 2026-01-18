from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def LoadAuxiliaryVersionsConfigsFromYamlFile(file_contents):
    """Convert Input YAML file into auxiliary versions configurations map.

  Args:
    file_contents: The YAML file contents of the file containing the auxiliary
      versions configurations.

  Returns:
    The auxiliary versions configuration mapping with service name as the key
    and config as the value.
  """
    aux_versions = {}
    for aux_config in file_contents:
        aux_versions[aux_config['name']] = {'version': aux_config['version']}
        if 'config_overrides' in aux_config:
            aux_versions[aux_config['name']]['configOverrides'] = aux_config['config_overrides']
    return aux_versions