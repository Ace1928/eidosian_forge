from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def RegistrationAttributeConfig():
    return concepts.ResourceParameterAttributeConfig(name='registration', help_text='The domain registration for the {resource}.')