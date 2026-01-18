from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesGetRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesGetRequest object.

  Fields:
    name: Required. Format:
      `projects/{project}/locations/{location}/instances/{instance}`.
  """
    name = _messages.StringField(1, required=True)