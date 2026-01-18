from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServicePerimetersResponse(_messages.Message):
    """A response to `ListServicePerimetersRequest`.

  Fields:
    nextPageToken: The pagination token to retrieve the next page of results.
      If the value is empty, no further results remain.
    servicePerimeters: List of the Service Perimeter instances.
  """
    nextPageToken = _messages.StringField(1)
    servicePerimeters = _messages.MessageField('ServicePerimeter', 2, repeated=True)