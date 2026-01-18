from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsProvidersGetRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsProvidersGetRequest object.

  Fields:
    name: Required. The name of the provider to retrieve. Format: `locations/{
      location}/workforcePools/{workforce_pool_id}/providers/{provider_id}`
  """
    name = _messages.StringField(1, required=True)