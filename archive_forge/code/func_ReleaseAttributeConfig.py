from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def ReleaseAttributeConfig():
    """Creates the release resource attribute."""
    return concepts.ResourceParameterAttributeConfig(name='release', help_text='The release associated with the {resource}.')