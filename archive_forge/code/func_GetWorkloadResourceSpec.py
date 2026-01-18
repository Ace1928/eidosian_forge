from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
def GetWorkloadResourceSpec():
    return concepts.ResourceSpec('assuredworkloads.organizations.locations.workloads', resource_name='workload', workloadsId=WorkloadAttributeConfig(), locationsId=LocationAttributeConfig(), organizationsId=OrganizationAttributeConfig())