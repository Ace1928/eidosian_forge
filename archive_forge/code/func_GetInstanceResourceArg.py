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
def GetInstanceResourceArg(help_text):
    """Constructs and returns the Instance Resource Argument."""

    def GetInstanceResourceSpec():
        """Constructs and returns the Resource specification for Instance."""

        def InstanceAttributeConfig():
            return concepts.ResourceParameterAttributeConfig(name='instance', help_text=help_text)

        def LocationAttributeConfig():
            return concepts.ResourceParameterAttributeConfig(name='location', help_text='Google Cloud location of this environment https://cloud.google.com/compute/docs/regions-zones/#locations.', fallthroughs=[deps.PropertyFallthrough(properties.VALUES.notebooks.location)])
        return concepts.ResourceSpec('notebooks.projects.locations.instances', resource_name='instance', api_version='v2', instancesId=InstanceAttributeConfig(), locationsId=LocationAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, disable_auto_completers=False)
    return concept_parsers.ConceptParser.ForResource('instance', GetInstanceResourceSpec(), help_text, required=True)