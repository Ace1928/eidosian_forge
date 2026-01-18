from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetVpcResourceSpec():
    """Constructs and returns the Resource specification for VPC."""

    def VpcAttributeConfig():
        return concepts.ResourceParameterAttributeConfig(name='vpc', help_text='fully qualified name of the VPC Datastream will peer to.')
    return concepts.ResourceSpec('compute.networks', resource_name='vpc', network=VpcAttributeConfig(), project=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)