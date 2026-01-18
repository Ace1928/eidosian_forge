from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidationErrorsBeforeIngestion(_messages.Message):
    """Summary of validation errors that occurred during the Verification
  phase. Next ID: 3

  Fields:
    bucketErrors: Optional. Provides a summary of the bucket level error
      stats.
    projectErrors: Optional. Provides a summary of the project level error
      stats.
  """
    bucketErrors = _messages.MessageField('BucketErrors', 1)
    projectErrors = _messages.MessageField('ProjectErrors', 2)