from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesListRequest(_messages.Message):
    """A ServiceusageServicesListRequest object.

  Fields:
    filter: Only list services that conform to the given filter. The allowed
      filter strings are `state:ENABLED` and `state:DISABLED`.
    pageSize: Requested size of the next page of data. Requested page size
      cannot exceed 200. If not set, the default page size is 50.
    pageToken: Token identifying which result to start with, which is returned
      by a previous list call.
    parent: Parent to search for services on. An example name would be:
      `projects/123` where `123` is the project number.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)