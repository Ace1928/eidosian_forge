from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageNote(_messages.Message):
    """Basis describes the base image portion (Note) of the DockerImage
  relationship. Linked occurrences are derived from this or an equivalent
  image via: FROM Or an equivalent reference, e.g., a tag of the resource_url.

  Fields:
    fingerprint: Required. Immutable. The fingerprint of the base image.
    resourceUrl: Required. Immutable. The resource_url for the resource
      representing the basis of associated occurrence images.
  """
    fingerprint = _messages.MessageField('Fingerprint', 1)
    resourceUrl = _messages.StringField(2)