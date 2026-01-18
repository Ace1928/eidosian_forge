from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetEndpointResourceSpec():
    """Gets endpoint resource spec."""
    return concepts.ResourceSpec('servicedirectory.projects.locations.namespaces.services.endpoints', resource_name='endpoint', endpointsId=EndpointAttributeConfig(), servicesId=ServiceAttributeConfig(), namespacesId=NamespaceAttributeConfig(), locationsId=LocationAttributeConfig(), projectsId=ProjectAttributeConfig())