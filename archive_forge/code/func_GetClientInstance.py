from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetClientInstance(release_track=base.ReleaseTrack.GA, use_http=True):
    """Returns an instance of the Cloud Build client.

  Args:
    release_track: The desired value of the enum
      googlecloudsdk.calliope.base.ReleaseTrack.
    use_http: bool, True to create an http object for this client.

  Returns:
    base_api.BaseApiClient, An instance of the Cloud Build client.
  """
    return apis.GetClientInstance(_API_NAME, RELEASE_TRACK_TO_API_VERSION[release_track], no_http=not use_http)