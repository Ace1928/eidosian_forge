from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetVolumeAttributeConfig(positional=True):
    """Return the Volume Attribute Config for resources.

  Args:
    positional: boolean value describing whether volume attribute is conditional

  Returns:
    volume resource parameter attribute config for resource specs

  """
    if positional:
        fallthroughs = []
    else:
        fallthroughs = [deps.ArgFallthrough('--volume')]
    help_text = 'The instance of the {resource}' if positional else 'The volume of the {resource}'
    return concepts.ResourceParameterAttributeConfig(name='volume', fallthroughs=fallthroughs, help_text=help_text)