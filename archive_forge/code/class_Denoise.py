from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Denoise(_messages.Message):
    """Denoise preprocessing configuration. **Note:** This configuration is not
  supported.

  Fields:
    strength: Set strength of the denoise. Enter a value between 0 and 1. The
      higher the value, the smoother the image. 0 is no denoising. The default
      is 0.
    tune: Set the denoiser mode. The default is `standard`. Supported denoiser
      modes: - `standard` - `grain`
  """
    strength = _messages.FloatField(1)
    tune = _messages.StringField(2)