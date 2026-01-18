from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Deblock(_messages.Message):
    """Deblock preprocessing configuration. **Note:** This configuration is not
  supported.

  Fields:
    enabled: Enable deblocker. The default is `false`.
    strength: Set strength of the deblocker. Enter a value between 0 and 1.
      The higher the value, the stronger the block removal. 0 is no
      deblocking. The default is 0.
  """
    enabled = _messages.BooleanField(1)
    strength = _messages.FloatField(2)