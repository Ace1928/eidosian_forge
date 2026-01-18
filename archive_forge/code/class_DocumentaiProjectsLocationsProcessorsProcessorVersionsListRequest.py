from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsProcessorVersionsListRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsProcessorVersionsListRequest
  object.

  Fields:
    pageSize: The maximum number of processor versions to return. If
      unspecified, at most `10` processor versions will be returned. The
      maximum value is `20`. Values above `20` will be coerced to `20`.
    pageToken: We will return the processor versions sorted by creation time.
      The page token will point to the next processor version.
    parent: Required. The parent (project, location and processor) to list all
      versions. Format:
      `projects/{project}/locations/{location}/processors/{processor}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)