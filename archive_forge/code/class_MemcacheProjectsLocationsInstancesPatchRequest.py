from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheProjectsLocationsInstancesPatchRequest(_messages.Message):
    """A MemcacheProjectsLocationsInstancesPatchRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    name: Required. Unique name of the resource in this scope including
      project and location using the form:
      `projects/{project_id}/locations/{location_id}/instances/{instance_id}`
      Note: Memcached instances are managed and addressed at the regional
      level so `location_id` here refers to a Google Cloud region; however,
      users may choose which zones Memcached nodes should be provisioned in
      within an instance. Refer to zones field for more details.
    updateMask: Required. Mask of fields to update. * `displayName`
  """
    instance = _messages.MessageField('Instance', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)