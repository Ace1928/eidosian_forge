from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1beta1IntotoDetails(_messages.Message):
    """This corresponds to a signed in-toto link - it is made up of one or more
  signatures and the in-toto link itself. This is used for occurrences of a
  Grafeas in-toto note.

  Fields:
    signatures: A GrafeasV1beta1IntotoSignature attribute.
    signed: A Link attribute.
  """
    signatures = _messages.MessageField('GrafeasV1beta1IntotoSignature', 1, repeated=True)
    signed = _messages.MessageField('Link', 2)