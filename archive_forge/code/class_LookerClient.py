from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
class LookerClient(object):
    """Wrapper for looker API client and associated resources."""

    def __init__(self, release_track):
        api_version = VERSION_MAP[release_track]
        self.release_track = release_track
        self.looker_client = apis.GetClientInstance('looker', api_version)