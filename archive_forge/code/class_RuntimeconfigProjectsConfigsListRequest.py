from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsListRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsListRequest object.

  Fields:
    pageSize: Specifies the number of results to return per page. If there are
      fewer elements than the specified number, returns all elements.
    pageToken: Specifies a page token to use. Set `pageToken` to a
      `nextPageToken` returned by a previous list request to get the next page
      of results.
    parent: The [project ID](https://support.google.com/cloud/answer/6158840?h
      l=en&ref_topic=6158848) for this request, in the format
      `projects/[PROJECT_ID]`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)