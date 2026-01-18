from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetAppProfileRef(instance, app_profile):
    """Get a resource reference to an app profile."""
    return resources.REGISTRY.Parse(app_profile, params={'projectsId': properties.VALUES.core.project.GetOrFail, 'instancesId': instance}, collection='bigtableadmin.projects.instances.appProfiles')