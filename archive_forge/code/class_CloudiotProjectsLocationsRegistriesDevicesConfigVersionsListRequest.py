from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesDevicesConfigVersionsListRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesDevicesConfigVersionsListRequest
  object.

  Fields:
    name: Required. The name of the device. For example,
      `projects/p0/locations/us-central1/registries/registry0/devices/device0`
      or `projects/p0/locations/us-
      central1/registries/registry0/devices/{num_id}`.
    numVersions: The number of versions to list. Versions are listed in
      decreasing order of the version number. The maximum number of versions
      retained is 10. If this value is zero, it will return all the versions
      available.
  """
    name = _messages.StringField(1, required=True)
    numVersions = _messages.IntegerField(2, variant=_messages.Variant.INT32)