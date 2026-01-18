from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ObbFile(_messages.Message):
    """An opaque binary blob file to install on the device before the test
  starts.

  Fields:
    obb: Required. Opaque Binary Blob (OBB) file(s) to install on the device.
    obbFileName: Required. OBB file name which must conform to the format as
      specified by Android e.g. [main|patch].0300110.com.example.android.obb
      which will be installed into \\/Android/obb/\\/ on the device.
  """
    obb = _messages.MessageField('FileReference', 1)
    obbFileName = _messages.StringField(2)