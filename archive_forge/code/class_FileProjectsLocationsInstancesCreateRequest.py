from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesCreateRequest(_messages.Message):
    """A FileProjectsLocationsInstancesCreateRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    instanceId: Required. The name of the instance to create. The name must be
      unique for the specified project and location.
    parent: Required. The instance's project and location, in the format
      `projects/{project_id}/locations/{location}`. In Filestore, locations
      map to Google Cloud zones, for example **us-west1-b**.
  """
    instance = _messages.MessageField('Instance', 1)
    instanceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)