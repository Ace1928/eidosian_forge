from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def ConfigsAttributeConfig(config_fallthrough=False, global_fallthrough=False):
    """Create a config attribute in resource argument.

  Args:
    config_fallthrough: If set, enables fallthroughs for the config attribute
      using the value set in workstations/config.
    global_fallthrough: If set, enables global fallthroughs for the config
      attribute.

  Returns:
    Config resource argument parameter config
  """
    fallthroughs = []
    if config_fallthrough:
        fallthroughs.append(deps.PropertyFallthrough(properties.VALUES.workstations.config))
    if global_fallthrough:
        fallthroughs.append(deps.Fallthrough(lambda: '-', hint='default is all configs'))
    return concepts.ResourceParameterAttributeConfig(name='config', fallthroughs=fallthroughs, help_text='The config for the {resource}.')