from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayApiApiController(_messages.Message):
    """An API Controller enforces global rules related to API Methods and
  Consumers, such as API Key checking, quota enforcement, and collections of
  usage statistics.

  Fields:
    managedService: The name of a Google Managed Service (
      https://cloud.google.com/service-infrastructure/docs/glossary#managed).
  """
    managedService = _messages.StringField(1)