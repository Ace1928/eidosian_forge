from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def DeployPolicyAttributeConfig():
    """Creates the Deploy Policy resource attribute."""
    return concepts.ResourceParameterAttributeConfig(name='deploy_policy', help_text='The Deploy Policy associated with the {resource}.')