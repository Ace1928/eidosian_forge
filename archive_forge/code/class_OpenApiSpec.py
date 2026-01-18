from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OpenApiSpec(_messages.Message):
    """A collection of OpenAPI specification files.

  Fields:
    openApiFiles: Individual files.
  """
    openApiFiles = _messages.MessageField('ConfigFile', 1, repeated=True)