from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BulkInsertInstanceResourcePerInstanceProperties(_messages.Message):
    """Per-instance properties to be set on individual instances. To be
  extended in the future.

  Fields:
    hostname: Specifies the hostname of the instance. More details in:
      https://cloud.google.com/compute/docs/instances/custom-hostname-
      vm#naming_convention
    name: This field is only temporary. It will be removed. Do not use it.
  """
    hostname = _messages.StringField(1)
    name = _messages.StringField(2)