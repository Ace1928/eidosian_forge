from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import transport
from oauth2client import client
def GetEffectiveIamEndpoint():
    """Returns the effective IAM endpoint.

  (1) If the [api_endpoint_overrides/iamcredentials] property is explicitly set,
  return the property value.
  (2) Otherwise if [core/universe_domain] value is not default, return
  "https://iamcredentials.{universe_domain_value}/".
  (3) Otherise return "https://iamcredentials.googleapis.com/"

  Returns:
    str: The effective IAM endpoint.
  """
    if properties.VALUES.api_endpoint_overrides.iamcredentials.IsExplicitlySet():
        return properties.VALUES.api_endpoint_overrides.iamcredentials.Get()
    universe_domain_property = properties.VALUES.core.universe_domain
    if universe_domain_property.Get() != universe_domain_property.default:
        return IAM_ENDPOINT_GDU.replace('googleapis.com', universe_domain_property.Get())
    return IAM_ENDPOINT_GDU