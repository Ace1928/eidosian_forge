from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetLocationResource(location, project=None):
    if not project:
        project = properties.VALUES.core.project.Get(required=True)
    return resources.REGISTRY.Parse(location, params={'projectsId': project}, collection='notebooks.projects.locations')