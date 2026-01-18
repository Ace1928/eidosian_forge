from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetClientMessages(release_track, api_version_override=None):
    return apis.GetMessagesModule(_API_CLIENT_NAME, api_version_override or _API_CLIENT_VERSION_MAP[release_track])