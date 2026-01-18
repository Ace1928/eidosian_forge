from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesBackupsCreateRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesBackupsCreateRequest object.

  Fields:
    instanceBackup: A InstanceBackup resource to be passed as the request
      body.
    parent: Required. Format:
      projects/{project}/locations/{location}/instances/{instance}
  """
    instanceBackup = _messages.MessageField('InstanceBackup', 1)
    parent = _messages.StringField(2, required=True)