from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolParameterKVMatchSpec(_messages.Message):
    """Spec for tool parameter key value match metric.

  Fields:
    useStrictStringMatch: Optional. Whether to use STRCIT string match on
      parameter values.
  """
    useStrictStringMatch = _messages.BooleanField(1)