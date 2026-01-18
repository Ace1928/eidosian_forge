from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsListRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsListRequest object.

  Fields:
    pageSize: The maximum number of processors to return. If unspecified, at
      most `50` processors will be returned. The maximum value is `100`.
      Values above `100` will be coerced to `100`.
    pageToken: We will return the processors sorted by creation time. The page
      token will point to the next processor.
    parent: Required. The parent (project and location) which owns this
      collection of Processors. Format:
      `projects/{project}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)