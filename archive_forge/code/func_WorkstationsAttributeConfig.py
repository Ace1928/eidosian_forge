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
def WorkstationsAttributeConfig():
    """Create a workstation attribute in resource argument.

  Returns:
    Workstation resource argument parameter config
  """
    return concepts.ResourceParameterAttributeConfig(name='workstation', help_text='The workstation.')