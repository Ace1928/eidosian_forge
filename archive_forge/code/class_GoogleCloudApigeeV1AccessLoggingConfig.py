from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AccessLoggingConfig(_messages.Message):
    """Access logging configuration enables customers to ship the access logs
  from the tenant projects to their own project's cloud logging. The feature
  is at the instance level ad disabled by default. It can be enabled during
  CreateInstance or UpdateInstance.

  Fields:
    enabled: Optional. Boolean flag that specifies whether the customer access
      log feature is enabled.
    filter: Optional. Ship the access log entries that match the status_code
      defined in the filter. The status_code is the only expected/supported
      filter field. (Ex: status_code) The filter will parse it to the Common
      Expression Language semantics for expression evaluation to build the
      filter condition. (Ex: "filter": status_code >= 200 && status_code < 300
      )
  """
    enabled = _messages.BooleanField(1)
    filter = _messages.StringField(2)