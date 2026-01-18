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
def GetSnapshotAttributeConfig(positional=True):
    if positional:
        help_text = 'The instance of the {resource}'
    else:
        help_text = 'The snapshot of the {resource}'
    return concepts.ResourceParameterAttributeConfig('snapshot', help_text=help_text)