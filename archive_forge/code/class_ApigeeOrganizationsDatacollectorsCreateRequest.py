from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDatacollectorsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsDatacollectorsCreateRequest object.

  Fields:
    dataCollectorId: ID of the data collector. Overrides any ID in the data
      collector resource. Must be a string beginning with `dc_` that contains
      only letters, numbers, and underscores.
    googleCloudApigeeV1DataCollector: A GoogleCloudApigeeV1DataCollector
      resource to be passed as the request body.
    parent: Required. Name of the organization in which to create the data
      collector in the following format: `organizations/{org}`.
  """
    dataCollectorId = _messages.StringField(1)
    googleCloudApigeeV1DataCollector = _messages.MessageField('GoogleCloudApigeeV1DataCollector', 2)
    parent = _messages.StringField(3, required=True)