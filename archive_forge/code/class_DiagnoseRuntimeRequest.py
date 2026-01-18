from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiagnoseRuntimeRequest(_messages.Message):
    """Request for creating a notebook instance diagnostic file.

  Fields:
    diagnosticConfig: Required. Defines flags that are used to run the
      diagnostic tool
    timeoutMinutes: Optional. Maxmium amount of time in minutes before the
      operation times out.
  """
    diagnosticConfig = _messages.MessageField('DiagnosticConfig', 1)
    timeoutMinutes = _messages.IntegerField(2, variant=_messages.Variant.INT32)