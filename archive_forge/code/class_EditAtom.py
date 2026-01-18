from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EditAtom(_messages.Message):
    """Edit atom.

  Fields:
    endTimeOffset: End time in seconds for the atom, relative to the input
      file timeline. When `end_time_offset` is not specified, the `inputs` are
      used until the end of the atom.
    inputs: List of Input.key values identifying files that should be used in
      this atom. The listed `inputs` must have the same timeline.
    key: A unique key for this atom. Must be specified when using advanced
      mapping.
    startTimeOffset: Start time in seconds for the atom, relative to the input
      file timeline. The default is `0s`.
  """
    endTimeOffset = _messages.StringField(1)
    inputs = _messages.StringField(2, repeated=True)
    key = _messages.StringField(3)
    startTimeOffset = _messages.StringField(4)