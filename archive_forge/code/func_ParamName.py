from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParamName(self, attribute_name):
    """Gets the param name from attribute. Used for autocompleters."""
    if attribute_name not in self.attribute_to_params_map:
        raise ValueError('No param name found for attribute [{}]. Existing attributes are [{}]'.format(attribute_name, ', '.join(sorted(self.attribute_to_params_map.keys()))))
    return self.attribute_to_params_map[attribute_name]