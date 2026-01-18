from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResponseStatusCode(_messages.Message):
    """A status to accept. Either a status code class like "2xx", or an integer
  status code like "200".

  Enums:
    StatusClassValueValuesEnum: A class of status codes to accept.

  Fields:
    statusClass: A class of status codes to accept.
    statusValue: A status code to accept.
  """

    class StatusClassValueValuesEnum(_messages.Enum):
        """A class of status codes to accept.

    Values:
      STATUS_CLASS_UNSPECIFIED: Default value that matches no status codes.
      STATUS_CLASS_1XX: The class of status codes between 100 and 199.
      STATUS_CLASS_2XX: The class of status codes between 200 and 299.
      STATUS_CLASS_3XX: The class of status codes between 300 and 399.
      STATUS_CLASS_4XX: The class of status codes between 400 and 499.
      STATUS_CLASS_5XX: The class of status codes between 500 and 599.
      STATUS_CLASS_ANY: The class of all status codes.
    """
        STATUS_CLASS_UNSPECIFIED = 0
        STATUS_CLASS_1XX = 1
        STATUS_CLASS_2XX = 2
        STATUS_CLASS_3XX = 3
        STATUS_CLASS_4XX = 4
        STATUS_CLASS_5XX = 5
        STATUS_CLASS_ANY = 6
    statusClass = _messages.EnumField('StatusClassValueValuesEnum', 1)
    statusValue = _messages.IntegerField(2, variant=_messages.Variant.INT32)