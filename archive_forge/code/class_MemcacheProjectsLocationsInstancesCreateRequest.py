from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheProjectsLocationsInstancesCreateRequest(_messages.Message):
    """A MemcacheProjectsLocationsInstancesCreateRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    instanceId: Required. The logical name of the Memcached instance in the
      user project with the following restrictions: * Must contain only
      lowercase letters, numbers, and hyphens. * Must start with a letter. *
      Must be between 1-40 characters. * Must end with a number or a letter. *
      Must be unique within the user project / location. If any of the above
      are not met, the API raises an invalid argument error.
    parent: Required. The resource name of the instance location using the
      form: `projects/{project_id}/locations/{location_id}` where
      `location_id` refers to a GCP region
  """
    instance = _messages.MessageField('Instance', 1)
    instanceId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)