from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShieldedVmIdentityEntry(_messages.Message):
    """A Shielded Instance Identity Entry.

  Fields:
    ekCert: A PEM-encoded X.509 certificate. This field can be empty.
    ekPub: A PEM-encoded public key.
  """
    ekCert = _messages.StringField(1)
    ekPub = _messages.StringField(2)