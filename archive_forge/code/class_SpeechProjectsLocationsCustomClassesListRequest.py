from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsCustomClassesListRequest(_messages.Message):
    """A SpeechProjectsLocationsCustomClassesListRequest object.

  Fields:
    pageSize: Number of results per requests. A valid page_size ranges from 0
      to 100 inclusive. If the page_size is zero or unspecified, a page size
      of 5 will be chosen. If the page size exceeds 100, it will be coerced
      down to 100. Note that a call might return fewer results than the
      requested page size.
    pageToken: A page token, received from a previous ListCustomClasses call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to ListCustomClasses must match the call that
      provided the page token.
    parent: Required. The project and location of CustomClass resources to
      list. The expected format is `projects/{project}/locations/{location}`.
    showDeleted: Whether, or not, to show resources that have been deleted.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)