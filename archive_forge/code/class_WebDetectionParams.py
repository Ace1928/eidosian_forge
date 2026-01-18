from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebDetectionParams(_messages.Message):
    """Parameters for web detection request.

  Fields:
    includeGeoResults: This field has no effect on results.
  """
    includeGeoResults = _messages.BooleanField(1)