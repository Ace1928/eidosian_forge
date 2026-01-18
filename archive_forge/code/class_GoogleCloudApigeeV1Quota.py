from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Quota(_messages.Message):
    """Quota contains the essential parameters needed that can be applied on
  the resources, methods, API source combination associated with this API
  product. While Quota is optional, setting it prevents requests from
  exceeding the provisioned parameters.

  Fields:
    interval: Required. Time interval over which the number of request
      messages is calculated.
    limit: Required. Upper limit allowed for the time interval and time unit
      specified. Requests exceeding this limit will be rejected.
    timeUnit: Time unit defined for the `interval`. Valid values include
      `minute`, `hour`, `day`, or `month`. If `limit` and `interval` are
      valid, the default value is `hour`; otherwise, the default is null.
  """
    interval = _messages.StringField(1)
    limit = _messages.StringField(2)
    timeUnit = _messages.StringField(3)