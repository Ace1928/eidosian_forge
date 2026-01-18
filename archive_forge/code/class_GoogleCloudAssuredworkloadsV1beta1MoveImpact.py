from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1MoveImpact(_messages.Message):
    """Represents the impact of moving the asset to the target.

  Fields:
    detail: Explanation of the impact.
  """
    detail = _messages.StringField(1)