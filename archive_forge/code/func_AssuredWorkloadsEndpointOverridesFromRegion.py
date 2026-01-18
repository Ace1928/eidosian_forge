from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import re
from googlecloudsdk.api_lib.assured import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
@contextlib.contextmanager
def AssuredWorkloadsEndpointOverridesFromRegion(release_track, region):
    """Context manager to regionalize Assured endpoints using a provided region.

  Args:
    release_track: str, Release track of the command being called.
    region: str, Region to use for regionalizing the Assured endpoint.

  Yields:
    None.
  """
    used_endpoint = GetEffectiveAssuredWorkloadsEndpoint(release_track, region)
    old_endpoint = properties.VALUES.api_endpoint_overrides.assuredworkloads.Get()
    try:
        log.status.Print('Using endpoint [{}]'.format(used_endpoint))
        if region:
            properties.VALUES.api_endpoint_overrides.assuredworkloads.Set(used_endpoint)
        yield
    finally:
        old_endpoint = properties.VALUES.api_endpoint_overrides.assuredworkloads.Set(old_endpoint)