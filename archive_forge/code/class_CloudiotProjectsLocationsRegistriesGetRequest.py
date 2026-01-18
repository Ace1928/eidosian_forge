from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesGetRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesGetRequest object.

  Fields:
    name: Required. The name of the device registry. For example,
      `projects/example-project/locations/us-central1/registries/my-registry`.
  """
    name = _messages.StringField(1, required=True)