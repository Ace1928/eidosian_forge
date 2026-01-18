from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplicationInfo(_messages.Message):
    """High level information corresponding to an application.

  Fields:
    applicationId: A string attribute.
    attempts: A ApplicationAttemptInfo attribute.
    coresGranted: A integer attribute.
    coresPerExecutor: A integer attribute.
    maxCores: A integer attribute.
    memoryPerExecutorMb: A integer attribute.
    name: A string attribute.
  """
    applicationId = _messages.StringField(1)
    attempts = _messages.MessageField('ApplicationAttemptInfo', 2, repeated=True)
    coresGranted = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    coresPerExecutor = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    maxCores = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    memoryPerExecutorMb = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    name = _messages.StringField(7)