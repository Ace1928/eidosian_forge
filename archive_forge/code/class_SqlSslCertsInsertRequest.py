from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlSslCertsInsertRequest(_messages.Message):
    """A SqlSslCertsInsertRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
    sslCertsInsertRequest: A SslCertsInsertRequest resource to be passed as
      the request body.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    sslCertsInsertRequest = _messages.MessageField('SslCertsInsertRequest', 3)