from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SwaggerSpec(_messages.Message):
    """A collection of swagger specification files.

  Fields:
    swaggerFiles: The individual files.
  """
    swaggerFiles = _messages.MessageField('File', 1, repeated=True)