from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def CustomTargetTypeAttributeConfig():
    """Creates the Custom Target Type resource attribute."""
    return concepts.ResourceParameterAttributeConfig(name='custom_target_type', help_text='The Custom Target Type associated with the {resource}.')