from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseAzureClusterResourceArg(args):
    return resources.REGISTRY.ParseRelativeName(args.CONCEPTS.cluster.Parse().RelativeName(), collection='gkemulticloud.projects.locations.azureClusters')