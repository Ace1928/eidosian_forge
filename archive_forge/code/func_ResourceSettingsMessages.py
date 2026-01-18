from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def ResourceSettingsMessages():
    """Returns the messages module for the Resource Settings service."""
    return apis.GetMessagesModule(RESOURCE_SETTINGS_API_NAME, RESOURCE_SETTINGS_API_VERSION)