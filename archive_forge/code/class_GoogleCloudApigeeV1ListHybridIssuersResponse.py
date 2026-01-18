from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListHybridIssuersResponse(_messages.Message):
    """A GoogleCloudApigeeV1ListHybridIssuersResponse object.

  Fields:
    issuers: Lists of hybrid services and its trusted issuer email ids.
  """
    issuers = _messages.MessageField('GoogleCloudApigeeV1ServiceIssuersMapping', 1, repeated=True)