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
def LocationsAttributeConfig(location_fallthrough=False, global_fallthrough=False):
    """Create a location attribute in resource argument.

  Args:
    location_fallthrough: If set, enables fallthroughs for the location
      attribute.
    global_fallthrough: If set, enables global fallthroughs for the location
      attribute.

  Returns:
    Location resource argument parameter config
  """
    fallthroughs = []
    if location_fallthrough:
        fallthroughs.append(deps.PropertyFallthrough(properties.VALUES.workstations.region))
    if global_fallthrough:
        fallthroughs.append(deps.Fallthrough(lambda: '-', hint='default is all regions'))
    return concepts.ResourceParameterAttributeConfig(name='region', fallthroughs=fallthroughs, help_text='The region for the {resource}.')