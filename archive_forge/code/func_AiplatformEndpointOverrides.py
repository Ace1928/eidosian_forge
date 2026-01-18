from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves.urllib import parse
@contextlib.contextmanager
def AiplatformEndpointOverrides(version, region, is_prediction=False):
    """Context manager to override the AI Platform endpoints for a while.

  Raises an error if
  region is not set.

  Args:
    version: str, implies the version that the endpoint will use.
    region: str, region of the AI Platform stack.
    is_prediction: bool, it's for prediction endpoint or not.

  Yields:
    None
  """
    used_endpoint = GetEffectiveEndpoint(version=version, region=region, is_prediction=is_prediction)
    log.status.Print('Using endpoint [{}]'.format(used_endpoint))
    properties.VALUES.api_endpoint_overrides.aiplatform.Set(used_endpoint)
    yield