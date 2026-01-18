from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyticsQuery(_messages.Message):
    """The configuration of a query to be run by QueryData or QueryDataLocal,
  or validated by ValidateQuery or ValidateQueryLocal.

  Fields:
    querySteps: Required. The query steps to execute. Each query step will
      correspond to a handle in the result proto.
  """
    querySteps = _messages.MessageField('QueryStep', 1, repeated=True)