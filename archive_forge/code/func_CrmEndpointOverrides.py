from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from six.moves.urllib import parse
@contextlib.contextmanager
def CrmEndpointOverrides(location):
    """Context manager to override the current CRM endpoint.

  The new endpoint will temporarily be the one corresponding to the given
  location.

  Args:
    location: str, location of the CRM backend (e.g. a cloud region or zone).
      Can be None to indicate global.

  Yields:
    None.
  """
    endpoint_property = getattr(properties.VALUES.api_endpoint_overrides, CRM_API_NAME)
    old_endpoint = endpoint_property.Get()
    is_staging_env = old_endpoint and (CRM_STAGING_REGIONAL_SUFFIX in old_endpoint or CRM_STAGING_GLOBAL_API == old_endpoint)
    try:
        if location and location != 'global':
            if is_staging_env:
                location = LOCATION_MAPPING.get(location, location)
                endpoint_property.Set(_DeriveCrmRegionalEndpoint('https://' + CRM_STAGING_REGIONAL_SUFFIX, location.replace('-', '')))
            else:
                endpoint_property.Set(_GetEffectiveCrmEndpoint(location))
        elif is_staging_env:
            endpoint_property.Set(CRM_STAGING_GLOBAL_API)
        yield
    finally:
        endpoint_property.Set(old_endpoint)