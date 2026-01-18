from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.compute.networks import flags as compute_network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as compute_subnet_flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.workbench import completers
from googlecloudsdk.core import properties
def AddNetworkArgument(help_text, parser):
    """Adds Resource arg for network to the parser."""

    def GetNetworkResourceSpec():
        """Constructs and returns the Resource specification for Subnet."""

        def NetworkAttributeConfig():
            return concepts.ResourceParameterAttributeConfig(name='network', help_text=help_text, completer=compute_network_flags.NetworksCompleter)
        return concepts.ResourceSpec('compute.networks', resource_name='network', network=NetworkAttributeConfig(), project=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)
    concept_parsers.ConceptParser.ForResource('--network', GetNetworkResourceSpec(), help_text).AddToParser(parser)