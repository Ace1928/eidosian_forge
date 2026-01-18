from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetAdminClusterResource(admin_cluster_name):
    relative_name = admin_cluster_name
    if admin_cluster_name.startswith('//'):
        parts = admin_cluster_name.split('/')
        relative_name = '/'.join(parts[3:])
    return resources.REGISTRY.ParseRelativeName(relative_name, collection='gkeonprem.projects.locations.vmwareAdminClusters')