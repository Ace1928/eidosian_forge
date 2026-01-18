from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.immersive_stream.xr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def ContentAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='name', help_text='Immersive Stream for XR content resource served by the {resource}.')