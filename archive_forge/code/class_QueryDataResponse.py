from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryDataResponse(_messages.Message):
    """The response data from QueryData.

  Fields:
    queryStepHandles: Handles to each of the query steps described in the
      request, excluding those for which the output_not_required flag was set.
      These may be passed to ReadQueryResults or used in a HandleQueryStep in
      a subsequent call to QueryData.
    restrictionConflicts: Conflicts between the query and the restrictions
      that were requested. Any restrictions present here were ignored when
      executing the query.
  """
    queryStepHandles = _messages.StringField(1, repeated=True)
    restrictionConflicts = _messages.MessageField('QueryRestrictionConflict', 2, repeated=True)