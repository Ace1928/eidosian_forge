from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExportFeatureValuesRequest(_messages.Message):
    """Request message for FeaturestoreService.ExportFeatureValues.

  Fields:
    destination: Required. Specifies destination location and format.
    featureSelector: Required. Selects Features to export values of.
    fullExport: Exports all historical values of all entities of the
      EntityType within a time range
    settings: Per-Feature export settings.
    snapshotExport: Exports the latest Feature values of all entities of the
      EntityType within a time range.
  """
    destination = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureValueDestination', 1)
    featureSelector = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureSelector', 2)
    fullExport = _messages.MessageField('GoogleCloudAiplatformV1beta1ExportFeatureValuesRequestFullExport', 3)
    settings = _messages.MessageField('GoogleCloudAiplatformV1beta1DestinationFeatureSetting', 4, repeated=True)
    snapshotExport = _messages.MessageField('GoogleCloudAiplatformV1beta1ExportFeatureValuesRequestSnapshotExport', 5)