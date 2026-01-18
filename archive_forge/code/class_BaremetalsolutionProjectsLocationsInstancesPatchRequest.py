from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsInstancesPatchRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsInstancesPatchRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    name: Immutable. The resource name of this `Instance`. Resource names are
      schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. Format:
      `projects/{project}/locations/{location}/instances/{instance}`
    updateMask: The list of fields to update. The currently supported fields
      are: `labels` `hyperthreading_enabled` `os_image` `ssh_keys`
      `kms_key_version`
  """
    instance = _messages.MessageField('Instance', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)