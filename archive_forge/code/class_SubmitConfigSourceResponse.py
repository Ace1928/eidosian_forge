from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubmitConfigSourceResponse(_messages.Message):
    """Response message for SubmitConfigSource method.

  Fields:
    diagnostics: Diagnostics occurred during config conversion.
    serviceConfig: The generated service configuration.
  """
    diagnostics = _messages.MessageField('Diagnostic', 1, repeated=True)
    serviceConfig = _messages.MessageField('Service', 2)