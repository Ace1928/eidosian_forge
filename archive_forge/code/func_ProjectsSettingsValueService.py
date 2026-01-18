from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def ProjectsSettingsValueService():
    """Returns the service class for the Project Settings Value resource."""
    client = ResourceSettingsClient()
    return client.projects_settings_value