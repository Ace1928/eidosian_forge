from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesAssetsCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesAssetsCreateRequest object.

  Fields:
    assetId: Required. Asset identifier. This ID will be used to generate
      names such as table names when publishing metadata to Hive Metastore and
      BigQuery. * Must contain only lowercase letters, numbers and hyphens. *
      Must start with a letter. * Must end with a number or a letter. * Must
      be between 1-63 characters. * Must be unique within the zone.
    googleCloudDataplexV1Asset: A GoogleCloudDataplexV1Asset resource to be
      passed as the request body.
    parent: Required. The resource name of the parent zone: projects/{project_
      number}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    assetId = _messages.StringField(1)
    googleCloudDataplexV1Asset = _messages.MessageField('GoogleCloudDataplexV1Asset', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)