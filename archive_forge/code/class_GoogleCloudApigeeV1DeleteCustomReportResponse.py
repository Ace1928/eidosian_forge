from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeleteCustomReportResponse(_messages.Message):
    """A GoogleCloudApigeeV1DeleteCustomReportResponse object.

  Fields:
    message: The response contains only a message field.
  """
    message = _messages.StringField(1)