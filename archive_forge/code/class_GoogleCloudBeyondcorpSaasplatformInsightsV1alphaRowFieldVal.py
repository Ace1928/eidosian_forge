from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformInsightsV1alphaRowFieldVal(_messages.Message):
    """Column or key value pair from the request as part of key to use in query
  or a single pair of the fetch response.

  Fields:
    displayName: Output only. Name of the field.
    filterAlias: Output only. Field name to be used in filter while requesting
      configured insight filtered on this field.
    id: Output only. Field id.
    value: Output only. Value of the field in string format. Acceptable values
      are strings or numbers.
  """
    displayName = _messages.StringField(1)
    filterAlias = _messages.StringField(2)
    id = _messages.StringField(3)
    value = _messages.StringField(4)