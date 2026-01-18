from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformInsightsV1alphaRow(_messages.Message):
    """Row of the fetch response consisting of a set of entries.

  Fields:
    fieldValues: Output only. Columns/entries/key-vals in the result.
  """
    fieldValues = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformInsightsV1alphaRowFieldVal', 1, repeated=True)