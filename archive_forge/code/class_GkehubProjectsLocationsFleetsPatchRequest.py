from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsFleetsPatchRequest(_messages.Message):
    """A GkehubProjectsLocationsFleetsPatchRequest object.

  Fields:
    fleet: A Fleet resource to be passed as the request body.
    name: Output only. The full, unique resource name of this fleet in the
      format of `projects/{project}/locations/{location}/fleets/{fleet}`. Each
      Google Cloud project can have at most one fleet resource, named
      "default".
    updateMask: Required. The fields to be updated;
  """
    fleet = _messages.MessageField('Fleet', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)