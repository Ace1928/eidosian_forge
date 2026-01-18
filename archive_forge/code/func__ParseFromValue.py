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
def _ParseFromValue(self, attribute_to_args_map, base_fallthroughs_map, parsed_args, allow_empty=False):
    """Helper for parsing a singular resource from user input."""
    fallthroughs_map = self.BuildFullFallthroughsMap(attribute_to_args_map, base_fallthroughs_map, parsed_args)
    try:
        return self.Initialize(fallthroughs_map, parsed_args=parsed_args)
    except InitializationError:
        if allow_empty:
            return None
        raise