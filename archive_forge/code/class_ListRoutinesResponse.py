from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRoutinesResponse(_messages.Message):
    """Describes the format of a single result page when listing routines.

  Fields:
    nextPageToken: A token to request the next page of results.
    routines: Routines in the requested dataset. Unless read_mask is set in
      the request, only the following fields are populated: etag, project_id,
      dataset_id, routine_id, routine_type, creation_time, last_modified_time,
      language, and remote_function_options.
  """
    nextPageToken = _messages.StringField(1)
    routines = _messages.MessageField('Routine', 2, repeated=True)