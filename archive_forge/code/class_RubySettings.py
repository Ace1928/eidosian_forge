from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RubySettings(_messages.Message):
    """Settings for Ruby client libraries.

  Fields:
    common: Some settings.
  """
    common = _messages.MessageField('CommonLanguageSettings', 1)