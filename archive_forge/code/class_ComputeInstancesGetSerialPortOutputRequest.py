from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstancesGetSerialPortOutputRequest(_messages.Message):
    """A ComputeInstancesGetSerialPortOutputRequest object.

  Fields:
    instance: Name of the instance for this request.
    port: Specifies which COM or serial port to retrieve data from.
    project: Project ID for this request.
    start: Specifies the starting byte position of the output to return. To
      start with the first byte of output to the specified port, omit this
      field or set it to `0`. If the output for that byte position is
      available, this field matches the `start` parameter sent with the
      request. If the amount of serial console output exceeds the size of the
      buffer (1 MB), the oldest output is discarded and is no longer
      available. If the requested start position refers to discarded output,
      the start position is adjusted to the oldest output still available, and
      the adjusted start position is returned as the `start` property value.
      You can also provide a negative start position, which translates to the
      most recent number of bytes written to the serial port. For example, -3
      is interpreted as the most recent 3 bytes written to the serial console.
    zone: The name of the zone for this request.
  """
    instance = _messages.StringField(1, required=True)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32, default=1)
    project = _messages.StringField(3, required=True)
    start = _messages.IntegerField(4)
    zone = _messages.StringField(5, required=True)