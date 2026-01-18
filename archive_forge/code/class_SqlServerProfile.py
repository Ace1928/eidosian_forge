from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerProfile(_messages.Message):
    """SQLServer database profile

  Fields:
    database: Required. Database for the SQLServer connection.
    hostname: Required. Hostname for the SQLServer connection.
    password: Required. Password for the SQLServer connection.
    port: Port for the SQLServer connection, default value is 1433.
    username: Required. Username for the SQLServer connection.
  """
    database = _messages.StringField(1)
    hostname = _messages.StringField(2)
    password = _messages.StringField(3)
    port = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    username = _messages.StringField(5)