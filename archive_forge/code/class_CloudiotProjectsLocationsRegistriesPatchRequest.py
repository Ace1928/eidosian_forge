from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesPatchRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesPatchRequest object.

  Fields:
    deviceRegistry: A DeviceRegistry resource to be passed as the request
      body.
    name: The resource path name. For example, `projects/example-
      project/locations/us-central1/registries/my-registry`.
    updateMask: Required. Only updates the `device_registry` fields indicated
      by this mask. The field mask must not be empty, and it must not contain
      fields that are immutable or only set by the server. Mutable top-level
      fields: `event_notification_config`, `http_config`, `mqtt_config`, and
      `state_notification_config`.
  """
    deviceRegistry = _messages.MessageField('DeviceRegistry', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)