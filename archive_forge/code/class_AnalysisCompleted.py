from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalysisCompleted(_messages.Message):
    """Indicates which analysis completed successfully. Multiple types of
  analysis can be performed on a single resource.

  Fields:
    analysisType: A string attribute.
  """
    analysisType = _messages.StringField(1, repeated=True)