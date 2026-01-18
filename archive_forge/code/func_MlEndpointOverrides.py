from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
@contextlib.contextmanager
def MlEndpointOverrides(region=None):
    """Context manager to override the AI Platform endpoints for a while.

  Args:
    region: str, region of the AI Platform stack.

  Yields:
    None.
  """
    used_endpoint = GetEffectiveMlEndpoint(region)
    old_endpoint = properties.VALUES.api_endpoint_overrides.ml.Get()
    try:
        log.status.Print('Using endpoint [{}]'.format(used_endpoint))
        if region and region != 'global':
            properties.VALUES.api_endpoint_overrides.ml.Set(used_endpoint)
        yield
    finally:
        old_endpoint = properties.VALUES.api_endpoint_overrides.ml.Set(old_endpoint)