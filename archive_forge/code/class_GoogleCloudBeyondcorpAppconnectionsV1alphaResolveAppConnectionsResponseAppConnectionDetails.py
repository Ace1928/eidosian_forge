from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectionsV1alphaResolveAppConnectionsResponseAppConnectionDetails(_messages.Message):
    """Details of the AppConnection.

  Fields:
    appConnection: A BeyondCorp AppConnection in the project.
    recentMigVms: If type=GCP_REGIONAL_MIG, contains most recent VM instances,
      like `https://www.googleapis.com/compute/v1/projects/{project_id}/zones/
      {zone_id}/instances/{instance_id}`.
  """
    appConnection = _messages.MessageField('GoogleCloudBeyondcorpAppconnectionsV1alphaAppConnection', 1)
    recentMigVms = _messages.StringField(2, repeated=True)