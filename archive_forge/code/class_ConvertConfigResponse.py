from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConvertConfigResponse(_messages.Message):
    """Response message for `ConvertConfig` method.

  Fields:
    diagnostics: Any errors or warnings that occured during config conversion.
    serviceConfig: The service configuration. Not set if errors occured during
      conversion.
  """
    diagnostics = _messages.MessageField('Diagnostic', 1, repeated=True)
    serviceConfig = _messages.MessageField('Service', 2)