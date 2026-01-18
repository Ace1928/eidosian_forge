from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisCreateRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisCreateRequest object.

  Fields:
    api: A Api resource to be passed as the request body.
    apiId: Required. The ID to use for the API, which will become the final
      component of the API's resource name. This value should be 4-63
      characters, and valid characters are /a-z-/. Following AIP-162, IDs must
      not have the form of a UUID.
    parent: Required. The parent, which owns this collection of APIs. Format:
      `projects/*/locations/*`
  """
    api = _messages.MessageField('Api', 1)
    apiId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)