from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1BuiltImage(_messages.Message):
    """An image built by the pipeline.

  Fields:
    digest: Docker Registry 2.0 digest.
    name: Name used to push the container image to Google Container Registry,
      as presented to `docker push`.
    pushTiming: Output only. Stores timing information for pushing the
      specified image.
  """
    digest = _messages.StringField(1)
    name = _messages.StringField(2)
    pushTiming = _messages.MessageField('GoogleDevtoolsCloudbuildV1TimeSpan', 3)