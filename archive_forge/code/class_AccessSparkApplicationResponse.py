from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSparkApplicationResponse(_messages.Message):
    """A summary of Spark Application

  Fields:
    application: Output only. High level information corresponding to an
      application.
  """
    application = _messages.MessageField('ApplicationInfo', 1)