from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlSslCertsCreateEphemeralRequest(_messages.Message):
    """A SqlSslCertsCreateEphemeralRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the Cloud SQL project.
    sslCertsCreateEphemeralRequest: A SslCertsCreateEphemeralRequest resource
      to be passed as the request body.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    sslCertsCreateEphemeralRequest = _messages.MessageField('SslCertsCreateEphemeralRequest', 3)