from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1APIProductAssociation(_messages.Message):
    """APIProductAssociation has the API product and its administrative state
  association.

  Fields:
    apiproduct: API product to be associated with the credential.
    status: The API product credential associated status. Valid values are
      `approved` or `revoked`.
  """
    apiproduct = _messages.StringField(1)
    status = _messages.StringField(2)