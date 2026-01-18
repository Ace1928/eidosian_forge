from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Derived(_messages.Message):
    """Derived describes the derived image portion (Occurrence) of the
  DockerImage relationship. This image would be produced from a Dockerfile
  with FROM .

  Fields:
    baseResourceUrl: Output only. This contains the base image URL for the
      derived image occurrence.
    distance: Output only. The number of layers by which this image differs
      from the associated image basis.
    fingerprint: Required. The fingerprint of the derived image.
    layerInfo: This contains layer-specific metadata, if populated it has
      length "distance" and is ordered with [distance] being the layer
      immediately following the base image and [1] being the final layer.
  """
    baseResourceUrl = _messages.StringField(1)
    distance = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    fingerprint = _messages.MessageField('Fingerprint', 3)
    layerInfo = _messages.MessageField('Layer', 4, repeated=True)