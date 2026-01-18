from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalDpaMoveList(_messages.Message):
    """An entry in a DPA's move list.

  Fields:
    dpaId: The ID of the DPA.
    frequencyRange: The frequency range that the move list affects.
  """
    dpaId = _messages.StringField(1)
    frequencyRange = _messages.MessageField('SasPortalFrequencyRange', 2)