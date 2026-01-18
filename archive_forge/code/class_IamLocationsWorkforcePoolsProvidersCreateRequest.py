from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsProvidersCreateRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsProvidersCreateRequest object.

  Fields:
    parent: Required. The pool to create this provider in. Format:
      `locations/{location}/workforcePools/{workforce_pool_id}`
    workforcePoolProvider: A WorkforcePoolProvider resource to be passed as
      the request body.
    workforcePoolProviderId: Required. The ID for the provider, which becomes
      the final component of the resource name. This value must be 4-32
      characters, and may contain the characters [a-z0-9-]. The prefix `gcp-`
      is reserved for use by Google, and may not be specified.
  """
    parent = _messages.StringField(1, required=True)
    workforcePoolProvider = _messages.MessageField('WorkforcePoolProvider', 2)
    workforcePoolProviderId = _messages.StringField(3)