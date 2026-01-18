from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetRegionAttributeConfig():
    return concepts.ResourceParameterAttributeConfig('region', 'The region of the {resource}.')