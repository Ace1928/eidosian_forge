from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsOperationsListRequest(_messages.Message):
    """A RunProjectsLocationsOperationsListRequest object.

  Fields:
    filter: Optional. A filter for matching the completed or in-progress
      operations. The supported formats of *filter* are: To query for only
      completed operations: done:true To query for only ongoing operations:
      done:false Must be empty to query for all of the latest operations for
      the given parent project.
    name: Required. To query for all of the operations for a project.
    pageSize: The maximum number of records that should be returned. Requested
      page size cannot exceed 100. If not set or set to less than or equal to
      0, the default page size is 100. .
    pageToken: Token identifying which result to start with, which is returned
      by a previous list call.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)