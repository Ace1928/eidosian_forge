from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def EkmConnectionAttributeConfig(kms_prefix=True):
    name = 'kms-ekmconnection' if kms_prefix else 'ekmconnection'
    return concepts.ResourceParameterAttributeConfig(name=name, help_text='The KMS ekm connection of the {resource}.')