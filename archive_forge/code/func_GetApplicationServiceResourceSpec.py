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
def GetApplicationServiceResourceSpec(arg_name='service', help_text='Name for the service'):
    """Constructs and returns the Resource specification for Application Service."""

    def ApplicationServiceAttributeConfig():
        return concepts.ResourceParameterAttributeConfig(name=arg_name, help_text=help_text)
    return concepts.ResourceSpec('apphub.projects.locations.applications.services', resource_name='service', servicesId=ApplicationServiceAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LocationAttributeConfig(), applicationsId=ApplicationResourceAttributeConfig(arg_name='application', help_text='Name for the application'))