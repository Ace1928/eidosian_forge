from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FluencyInput(_messages.Message):
    """Input for fluency metric.

  Fields:
    instance: Required. Fluency instance.
    metricSpec: Required. Spec for fluency score metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1FluencyInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1FluencySpec', 2)