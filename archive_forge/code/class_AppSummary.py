from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppSummary(_messages.Message):
    """A AppSummary object.

  Fields:
    numCompletedJobs: A integer attribute.
    numCompletedStages: A integer attribute.
  """
    numCompletedJobs = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    numCompletedStages = _messages.IntegerField(2, variant=_messages.Variant.INT32)