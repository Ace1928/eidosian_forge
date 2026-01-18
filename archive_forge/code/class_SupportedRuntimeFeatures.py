from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedRuntimeFeatures(_messages.Message):
    """Supported runtime features of a connector version.

  Fields:
    actionApis: Specifies if the connector supports action apis like
      'executeAction'.
    entityApis: Specifies if the connector supports entity apis like
      'createEntity'.
    sqlQuery: Specifies if the connector supports 'ExecuteSqlQuery' operation.
  """
    actionApis = _messages.BooleanField(1)
    entityApis = _messages.BooleanField(2)
    sqlQuery = _messages.BooleanField(3)