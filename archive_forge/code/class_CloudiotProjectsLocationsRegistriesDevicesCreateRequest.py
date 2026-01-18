from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesDevicesCreateRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesDevicesCreateRequest object.

  Fields:
    device: A Device resource to be passed as the request body.
    parent: Required. The name of the device registry where this device should
      be created. For example, `projects/example-project/locations/us-
      central1/registries/my-registry`.
  """
    device = _messages.MessageField('Device', 1)
    parent = _messages.StringField(2, required=True)