from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalFrequencyRange(_messages.Message):
    """Frequency range from `low_frequency` to `high_frequency`.

  Fields:
    highFrequencyMhz: The highest frequency of the frequency range in MHz.
    lowFrequencyMhz: The lowest frequency of the frequency range in MHz.
  """
    highFrequencyMhz = _messages.FloatField(1)
    lowFrequencyMhz = _messages.FloatField(2)