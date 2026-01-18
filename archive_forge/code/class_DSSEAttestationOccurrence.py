from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DSSEAttestationOccurrence(_messages.Message):
    """Deprecated. Prefer to use a regular Occurrence, and populate the
  Envelope at the top level of the Occurrence.

  Fields:
    envelope: If doing something security critical, make sure to verify the
      signatures in this metadata.
    statement: A InTotoStatement attribute.
  """
    envelope = _messages.MessageField('Envelope', 1)
    statement = _messages.MessageField('InTotoStatement', 2)