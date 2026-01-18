from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ZoneDiscoverySpecJsonOptions(_messages.Message):
    """Describe JSON data format.

  Fields:
    disableTypeInference: Optional. Whether to disable the inference of data
      type for Json data. If true, all columns will be registered as their
      primitive types (strings, number or boolean).
    encoding: Optional. The character encoding of the data. The default is
      UTF-8.
  """
    disableTypeInference = _messages.BooleanField(1)
    encoding = _messages.StringField(2)