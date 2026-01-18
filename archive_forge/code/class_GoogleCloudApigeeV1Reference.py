from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Reference(_messages.Message):
    """A Reference configuration. References must refer to a keystore that also
  exists in the parent environment.

  Fields:
    description: Optional. A human-readable description of this reference.
    name: Required. The resource id of this reference. Values must match the
      regular expression [\\w\\s\\-.]+.
    refers: Required. The id of the resource to which this reference refers.
      Must be the id of a resource that exists in the parent environment and
      is of the given resource_type.
    resourceType: The type of resource referred to by this reference. Valid
      values are 'KeyStore' or 'TrustStore'.
  """
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    refers = _messages.StringField(3)
    resourceType = _messages.StringField(4)