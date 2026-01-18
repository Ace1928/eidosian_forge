from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CoherenceInput(_messages.Message):
    """Input for coherence metric.

  Fields:
    instance: Required. Coherence instance.
    metricSpec: Required. Spec for coherence score metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1CoherenceInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1CoherenceSpec', 2)