from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UrlMapsValidateResponse(_messages.Message):
    """A UrlMapsValidateResponse object.

  Fields:
    result: A UrlMapValidationResult attribute.
  """
    result = _messages.MessageField('UrlMapValidationResult', 1)