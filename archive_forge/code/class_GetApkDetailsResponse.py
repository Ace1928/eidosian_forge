from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GetApkDetailsResponse(_messages.Message):
    """Response containing the details of the specified Android application.

  Fields:
    apkDetail: Details of the Android App.
  """
    apkDetail = _messages.MessageField('ApkDetail', 1)