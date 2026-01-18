from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LatencyCriteria(_messages.Message):
    """Parameters for a latency threshold SLI.

  Fields:
    threshold: Good service is defined to be the count of requests made to
      this service that return in no more than threshold.
  """
    threshold = _messages.StringField(1)