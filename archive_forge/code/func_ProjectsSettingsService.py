from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def ProjectsSettingsService():
    """Returns the service class for the Project Settings resource."""
    client = ResourceSettingsClient()
    return client.projects_settings