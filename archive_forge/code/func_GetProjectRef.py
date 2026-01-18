from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetProjectRef():
    return resources.REGISTRY.Parse(None, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection=PROJECTS_COLLECTION)