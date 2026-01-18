from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetResourceAllowanceResourceSpec():
    return concepts.ResourceSpec('batch.projects.locations.resourceAllowances', api_version='v1alpha', resource_name='resourceAllowance', projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=LocationAttributeConfig(), resourceAllowancesId=ResourceAllowanceAttributeConfig())