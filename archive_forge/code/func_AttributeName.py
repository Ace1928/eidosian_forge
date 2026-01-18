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
def AttributeName(self, param_name):
    """Gets the attribute name from param name. Used for autocompleters."""
    for attribute_name, p in self.attribute_to_params_map.items():
        if p == param_name:
            return attribute_name
    else:
        return None