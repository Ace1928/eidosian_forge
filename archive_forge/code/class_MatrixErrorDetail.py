from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MatrixErrorDetail(_messages.Message):
    """Describes a single error or issue with a matrix.

  Fields:
    message: Output only. A human-readable message about how the error in the
      TestMatrix. Expands on the `reason` field with additional details and
      possible options to fix the issue.
    reason: Output only. The reason for the error. This is a constant value in
      UPPER_SNAKE_CASE that identifies the cause of the error.
  """
    message = _messages.StringField(1)
    reason = _messages.StringField(2)