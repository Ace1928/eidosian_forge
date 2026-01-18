from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisKeyvaluemapsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsApisKeyvaluemapsCreateRequest object.

  Fields:
    googleCloudApigeeV1KeyValueMap: A GoogleCloudApigeeV1KeyValueMap resource
      to be passed as the request body.
    parent: Required. Name of the environment in which to create the key value
      map. Use the following structure in your request:
      `organizations/{org}/apis/{api}`
  """
    googleCloudApigeeV1KeyValueMap = _messages.MessageField('GoogleCloudApigeeV1KeyValueMap', 1)
    parent = _messages.StringField(2, required=True)