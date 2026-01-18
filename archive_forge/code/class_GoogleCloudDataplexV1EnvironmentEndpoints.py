from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EnvironmentEndpoints(_messages.Message):
    """URI Endpoints to access sessions associated with the Environment.

  Fields:
    notebooks: Output only. URI to serve notebook APIs
    sql: Output only. URI to serve SQL APIs
  """
    notebooks = _messages.StringField(1)
    sql = _messages.StringField(2)