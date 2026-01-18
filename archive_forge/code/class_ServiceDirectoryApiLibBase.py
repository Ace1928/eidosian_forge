from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
class ServiceDirectoryApiLibBase(object):
    """The base class for all Service Directory clients."""

    def __init__(self, release_track=base.ReleaseTrack.GA):
        self.client = apis.GetClientInstance(_API_NAME, _VERSION_MAP.get(release_track))
        self.msgs = apis.GetMessagesModule(_API_NAME, _VERSION_MAP.get(release_track))