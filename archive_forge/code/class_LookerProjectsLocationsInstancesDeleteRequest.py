from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesDeleteRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesDeleteRequest object.

  Fields:
    force: Whether to force cascading delete.
    name: Required. Format:
      `projects/{project}/locations/{location}/instances/{instance}`.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)