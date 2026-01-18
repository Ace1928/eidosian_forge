from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DeviceFile(_messages.Message):
    """A single device file description.

  Fields:
    obbFile: A reference to an opaque binary blob file.
    regularFile: A reference to a regular file.
  """
    obbFile = _messages.MessageField('ObbFile', 1)
    regularFile = _messages.MessageField('RegularFile', 2)