from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPeeredDnsDomainsResponse(_messages.Message):
    """Response to list peered DNS domains for a given connection.

  Fields:
    peeredDnsDomains: The list of peered DNS domains.
  """
    peeredDnsDomains = _messages.MessageField('PeeredDnsDomain', 1, repeated=True)