from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAttributesRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAttributesRequest object.

  Fields:
    googleCloudApigeeV1Attributes: A GoogleCloudApigeeV1Attributes resource to
      be passed as the request body.
    parent: Required. Email address of the developer for which attributes are
      being updated. Use the following structure in your request:
      `organizations/{org}/developers/{developer_email}`
  """
    googleCloudApigeeV1Attributes = _messages.MessageField('GoogleCloudApigeeV1Attributes', 1)
    parent = _messages.StringField(2, required=True)