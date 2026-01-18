from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from six.moves.urllib import parse
@contextlib.contextmanager
def GkemulticloudEndpointOverride(location, track=base.ReleaseTrack.GA):
    """Context manager to override the GKE Multi-cloud endpoint temporarily.

  Args:
    location: str, location to use for GKE Multi-cloud.
    track: calliope_base.ReleaseTrack, Release track of the endpoint.

  Yields:
    None.
  """
    original_ep = properties.VALUES.api_endpoint_overrides.gkemulticloud.Get()
    try:
        if not original_ep:
            if not location:
                raise ValueError('A location must be specified.')
            _ValidateLocation(location)
            regional_ep = _GetEffectiveEndpoint(location, track=track)
            properties.VALUES.api_endpoint_overrides.gkemulticloud.Set(regional_ep)
        yield
    finally:
        if not original_ep:
            properties.VALUES.api_endpoint_overrides.gkemulticloud.Set(original_ep)