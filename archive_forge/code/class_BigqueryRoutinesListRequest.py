from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryRoutinesListRequest(_messages.Message):
    """A BigqueryRoutinesListRequest object.

  Fields:
    datasetId: Required. Dataset ID of the routines to list
    filter: If set, then only the Routines matching this filter are returned.
      The supported format is `routineType:{RoutineType}`, where
      `{RoutineType}` is a RoutineType enum. For example:
      `routineType:SCALAR_FUNCTION`.
    maxResults: The maximum number of results to return in a single response
      page. Leverage the page tokens to iterate through the entire collection.
    pageToken: Page token, returned by a previous call, to request the next
      page of results
    projectId: Required. Project ID of the routines to list
    readMask: If set, then only the Routine fields in the field mask, as well
      as project_id, dataset_id and routine_id, are returned in the response.
      If unset, then the following Routine fields are returned: etag,
      project_id, dataset_id, routine_id, routine_type, creation_time,
      last_modified_time, and language.
  """
    datasetId = _messages.StringField(1, required=True)
    filter = _messages.StringField(2)
    maxResults = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(4)
    projectId = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)