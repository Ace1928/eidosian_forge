from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PullStartedEvent(_messages.Message):
    """An event generated when the worker starts pulling an image.

  Fields:
    imageUri: The URI of the image that was pulled.
  """
    imageUri = _messages.StringField(1)