from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def TagMessages():
    """Returns the messages module for the Tags service."""
    return apis.GetMessagesModule('cloudresourcemanager', TAGS_API_VERSION)