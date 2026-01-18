from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformInsightsV1alphaInsightMetadataField(_messages.Message):
    """Field metadata. Commonly understandable name and description for the
  field. Multiple such fields constitute the Insight.

  Fields:
    description: Output only. Description of the field.
    displayName: Output only. Name of the field.
    filterAlias: Output only. Field name to be used in filter while requesting
      configured insight filtered on this field.
    filterable: Output only. Indicates whether the field can be used for
      filtering.
    groupable: Output only. Indicates whether the field can be used for
      grouping in custom grouping request.
    id: Output only. Field id for which this is the metadata.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    filterAlias = _messages.StringField(3)
    filterable = _messages.BooleanField(4)
    groupable = _messages.BooleanField(5)
    id = _messages.StringField(6)