from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GetGcloudProject(self):
    """Gets the current project if project is not specified."""
    return properties.VALUES.core.project.Get(required=True)