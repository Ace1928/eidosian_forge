from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Certificate(_messages.Message):
    """A GoogleCloudApigeeV1Certificate object.

  Fields:
    certInfo: Chain of certificates under this name.
  """
    certInfo = _messages.MessageField('GoogleCloudApigeeV1CertInfo', 1, repeated=True)