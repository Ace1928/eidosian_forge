from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HandleQueryStep(_messages.Message):
    """A query step that reads the results of a step in a previous query
  operation as its input.

  Fields:
    queryStepHandle: Required. A handle to a query step from a previous call
      to QueryData.
  """
    queryStepHandle = _messages.StringField(1)