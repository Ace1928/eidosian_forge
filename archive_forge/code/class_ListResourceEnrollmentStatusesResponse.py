from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListResourceEnrollmentStatusesResponse(_messages.Message):
    """Response message with all the descendent resources with enrollment.

  Fields:
    nextPageToken: Output only. The token to retrieve the next page of
      results.
    resourceEnrollmentStatuses: The resources with their enrollment status.
  """
    nextPageToken = _messages.StringField(1)
    resourceEnrollmentStatuses = _messages.MessageField('ResourceEnrollmentStatus', 2, repeated=True)