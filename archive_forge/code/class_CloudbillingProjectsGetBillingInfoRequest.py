from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingProjectsGetBillingInfoRequest(_messages.Message):
    """A CloudbillingProjectsGetBillingInfoRequest object.

  Fields:
    name: Required. The resource name of the project for which billing
      information is retrieved. For example, `projects/tokyo-rain-123`.
  """
    name = _messages.StringField(1, required=True)