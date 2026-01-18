from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetFaultResourceSpec(arg_name='fault', help_text=None):
    """Constructs and returns the Resource specification for Fault."""

    def FaultAttributeConfig():
        return concepts.ResourceParameterAttributeConfig(name=arg_name, help_text=help_text)
    return concepts.ResourceSpec('faultinjectiontesting.projects.locations.faults', resource_name='fault', faultsId=FaultAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LocationAttributeConfig())