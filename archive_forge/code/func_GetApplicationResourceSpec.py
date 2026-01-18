from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apphub import utils as apphub_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetApplicationResourceSpec(arg_name='application', help_text=None):
    """Constructs and returns the Resource specification for Application."""
    return concepts.ResourceSpec('apphub.projects.locations.applications', resource_name='application', applicationsId=ApplicationResourceAttributeConfig(arg_name, help_text), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LocationAttributeConfig())