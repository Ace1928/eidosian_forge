from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsLocationsKeysPatchRequest(_messages.Message):
    """A ApikeysProjectsLocationsKeysPatchRequest object.

  Fields:
    name: Output only. The resource name of the key. The `name` has the form:
      `projects//locations/global/keys/`. For example: `projects/123456867718/
      locations/global/keys/b7ff1f9f-8275-410a-94dd-3855ee9b5dd2` NOTE: Key is
      a global resource; hence the only supported value for location is
      `global`.
    updateMask: The field mask specifies which fields to be updated as part of
      this request. All other fields are ignored. Mutable fields are:
      `display_name`,`restrictions` and `annotations`. If an update mask is
      not provided, the service treats it as an implied mask equivalent to all
      allowed fields that are set on the wire. If the field mask has a special
      value "*", the service treats it equivalent to replace all allowed
      mutable fields.
    v2Key: A V2Key resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    v2Key = _messages.MessageField('V2Key', 3)