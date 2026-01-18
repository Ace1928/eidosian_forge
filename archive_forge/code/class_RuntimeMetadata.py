from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeMetadata(_messages.Message):
    """RuntimeMetadata describing a runtime environment.

  Fields:
    parameters: The parameters for the template.
    sdkInfo: SDK Info for the template.
  """
    parameters = _messages.MessageField('ParameterMetadata', 1, repeated=True)
    sdkInfo = _messages.MessageField('SDKInfo', 2)