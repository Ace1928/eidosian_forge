from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.immersive_stream.xr import api_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def UpdateFallbackUrl(release_track, instance_ref, fallback_url):
    """Update fallback url of an Immersive Stream for XR service instance.

  Args:
    release_track: ALPHA or GA release track
    instance_ref: resource object - service instance to be updated
    fallback_url: string - fallback url to redirect users to when the instance
      is not available

  Returns:
    An Operation object which can be used to check on the progress of the
    service instance update.
  """
    client = api_util.GetClient(release_track)
    messages = api_util.GetMessages(release_track)
    service = client.ProjectsLocationsStreamInstancesService(client)
    stream_config = messages.StreamConfig(fallbackUri=fallback_url)
    instance = messages.StreamInstance()
    instance.streamConfig = stream_config
    return service.Patch(messages.StreamProjectsLocationsStreamInstancesPatchRequest(name=instance_ref.RelativeName(), streamInstance=instance, updateMask='stream_config'))