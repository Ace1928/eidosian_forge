from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def FoldersSettingsService():
    """Returns the service class for the Folders Settings resource."""
    client = ResourceSettingsClient()
    return client.folders_settings