from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class VulnerableHeaders(_messages.Message):
    """Information about vulnerable or missing HTTP Headers.

  Fields:
    headers: List of vulnerable headers.
    missingHeaders: List of missing headers.
  """
    headers = _messages.MessageField('Header', 1, repeated=True)
    missingHeaders = _messages.MessageField('Header', 2, repeated=True)