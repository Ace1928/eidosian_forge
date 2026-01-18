from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FluencyInstance(_messages.Message):
    """Spec for fluency instance.

  Fields:
    prediction: Required. Output of the evaluated model.
  """
    prediction = _messages.StringField(1)