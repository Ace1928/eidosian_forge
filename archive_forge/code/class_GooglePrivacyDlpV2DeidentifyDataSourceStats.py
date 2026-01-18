from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DeidentifyDataSourceStats(_messages.Message):
    """Summary of what was modified during a transformation.

  Fields:
    transformationCount: Number of successfully applied transformations.
    transformationErrorCount: Number of errors encountered while trying to
      apply transformations.
    transformedBytes: Total size in bytes that were transformed in some way.
  """
    transformationCount = _messages.IntegerField(1)
    transformationErrorCount = _messages.IntegerField(2)
    transformedBytes = _messages.IntegerField(3)