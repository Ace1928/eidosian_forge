from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesDevicesStatesListRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesDevicesStatesListRequest object.

  Fields:
    name: Required. The name of the device. For example,
      `projects/p0/locations/us-central1/registries/registry0/devices/device0`
      or `projects/p0/locations/us-
      central1/registries/registry0/devices/{num_id}`.
    numStates: The number of states to list. States are listed in descending
      order of update time. The maximum number of states retained is 10. If
      this value is zero, it will return all the states available.
  """
    name = _messages.StringField(1, required=True)
    numStates = _messages.IntegerField(2, variant=_messages.Variant.INT32)