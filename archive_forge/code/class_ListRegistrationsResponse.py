from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRegistrationsResponse(_messages.Message):
    """Response for the `ListRegistrations` method.

  Fields:
    nextPageToken: When present, there are more results to retrieve. Set
      `page_token` to this value on a subsequent call to get the next page of
      results.
    registrations: A list of `Registration`s.
  """
    nextPageToken = _messages.StringField(1)
    registrations = _messages.MessageField('Registration', 2, repeated=True)