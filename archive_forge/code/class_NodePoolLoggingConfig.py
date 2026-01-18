from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodePoolLoggingConfig(_messages.Message):
    """NodePoolLoggingConfig specifies logging configuration for nodepools.

  Fields:
    variantConfig: Logging variant configuration.
  """
    variantConfig = _messages.MessageField('LoggingVariantConfig', 1)