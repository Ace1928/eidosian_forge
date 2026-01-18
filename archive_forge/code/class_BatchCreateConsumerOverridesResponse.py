from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateConsumerOverridesResponse(_messages.Message):
    """Response message for BatchCreateConsumerOverrides

  Fields:
    overrides: The overrides that were created.
  """
    overrides = _messages.MessageField('QuotaOverride', 1, repeated=True)