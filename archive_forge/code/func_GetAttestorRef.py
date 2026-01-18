from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetAttestorRef(attestor_name):
    return resources.REGISTRY.Parse(attestor_name, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection=PROJECTS_ATTESTORS_COLLECTION)