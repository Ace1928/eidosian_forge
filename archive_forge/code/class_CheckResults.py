from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckResults(_messages.Message):
    """Result of evaluating one or more checks.

  Fields:
    results: Per-check details.
  """
    results = _messages.MessageField('CheckResult', 1, repeated=True)