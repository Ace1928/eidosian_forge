from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IosRuntimeConfiguration(_messages.Message):
    """iOS configuration that can be selected at the time a test is run.

  Fields:
    locales: The set of available locales.
    orientations: The set of available orientations.
  """
    locales = _messages.MessageField('Locale', 1, repeated=True)
    orientations = _messages.MessageField('Orientation', 2, repeated=True)