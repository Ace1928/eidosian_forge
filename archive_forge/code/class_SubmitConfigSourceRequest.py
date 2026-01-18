from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubmitConfigSourceRequest(_messages.Message):
    """Request message for SubmitConfigSource method.

  Fields:
    configSource: The source configuration for the service.
    validateOnly: Optional. If set, this will result in the generation of a
      `google.api.Service` configuration based on the `ConfigSource` provided,
      but the generated config and the sources will NOT be persisted.
  """
    configSource = _messages.MessageField('ConfigSource', 1)
    validateOnly = _messages.BooleanField(2)