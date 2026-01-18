from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerEncryptionOptions(_messages.Message):
    """Encryption settings for the SQL Server database.

  Fields:
    certPath: Required. Path to certificate.
    pkvPassword: Optional. Input only. Private key password. To be deprecated
    pkvPath: Optional. Path to certificate private key. To be deprecated
    pvkPassword: Required. Input only. Private key password.
    pvkPath: Required. Path to certificate private key.
  """
    certPath = _messages.StringField(1)
    pkvPassword = _messages.StringField(2)
    pkvPath = _messages.StringField(3)
    pvkPassword = _messages.StringField(4)
    pvkPath = _messages.StringField(5)