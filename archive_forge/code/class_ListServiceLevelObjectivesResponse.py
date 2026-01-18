from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceLevelObjectivesResponse(_messages.Message):
    """The ListServiceLevelObjectives response.

  Fields:
    nextPageToken: If there are more results than have been returned, then
      this field is set to a non-empty value. To see the additional results,
      use that value as page_token in the next call to this method.
    serviceLevelObjectives: The ServiceLevelObjectives matching the specified
      filter.
  """
    nextPageToken = _messages.StringField(1)
    serviceLevelObjectives = _messages.MessageField('ServiceLevelObjective', 2, repeated=True)