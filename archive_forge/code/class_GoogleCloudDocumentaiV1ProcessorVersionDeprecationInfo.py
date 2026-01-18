from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ProcessorVersionDeprecationInfo(_messages.Message):
    """Information about the upcoming deprecation of this processor version.

  Fields:
    deprecationTime: The time at which this processor version will be
      deprecated.
    replacementProcessorVersion: If set, the processor version that will be
      used as a replacement.
  """
    deprecationTime = _messages.StringField(1)
    replacementProcessorVersion = _messages.StringField(2)