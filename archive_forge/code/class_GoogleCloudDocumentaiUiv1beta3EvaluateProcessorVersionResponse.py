from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3EvaluateProcessorVersionResponse(_messages.Message):
    """Response of the EvaluateProcessorVersion method.

  Fields:
    evaluation: The resource name of the created evaluation.
  """
    evaluation = _messages.StringField(1)