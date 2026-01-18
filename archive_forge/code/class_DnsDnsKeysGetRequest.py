from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsDnsKeysGetRequest(_messages.Message):
    """A DnsDnsKeysGetRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    digestType: An optional comma-separated list of digest types to compute
      and display for key signing keys. If omitted, the recommended digest
      type is computed and displayed.
    dnsKeyId: The identifier of the requested DnsKey.
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or ID.
    project: Identifies the project addressed by this request.
  """
    clientOperationId = _messages.StringField(1)
    digestType = _messages.StringField(2)
    dnsKeyId = _messages.StringField(3, required=True)
    managedZone = _messages.StringField(4, required=True)
    project = _messages.StringField(5, required=True)