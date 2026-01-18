from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesPatchRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesPatchRequest object.

  Fields:
    instance: A Instance resource to be passed as the request body.
    name: Output only. Format:
      `projects/{project}/locations/{location}/instances/{instance}`.
    updateMask: Required. Field mask used to specify the fields to be
      overwritten in the Instance resource by the update. The fields specified
      in the mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask.
  """
    instance = _messages.MessageField('Instance', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)