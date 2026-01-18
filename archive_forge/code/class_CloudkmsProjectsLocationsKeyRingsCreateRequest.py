from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCreateRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCreateRequest object.

  Fields:
    keyRing: A KeyRing resource to be passed as the request body.
    keyRingId: Required. It must be unique within a location and match the
      regular expression `[a-zA-Z0-9_-]{1,63}`
    parent: Required. The resource name of the location associated with the
      KeyRings, in the format `projects/*/locations/*`.
  """
    keyRing = _messages.MessageField('KeyRing', 1)
    keyRingId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)