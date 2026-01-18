from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesGetRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesGetRequest object.

  Fields:
    name: Required. The instance resource name in the format
      projects/{project}/locations/{location}/instances/{instance}.
  """
    name = _messages.StringField(1, required=True)