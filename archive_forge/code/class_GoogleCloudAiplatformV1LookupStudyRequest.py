from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1LookupStudyRequest(_messages.Message):
    """Request message for VizierService.LookupStudy.

  Fields:
    displayName: Required. The user-defined display name of the Study
  """
    displayName = _messages.StringField(1)