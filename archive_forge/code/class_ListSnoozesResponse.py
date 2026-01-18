from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSnoozesResponse(_messages.Message):
    """The results of a successful ListSnoozes call, containing the matching
  Snoozes.

  Fields:
    nextPageToken: Page token for repeated calls to ListSnoozes, to fetch
      additional pages of results. If this is empty or missing, there are no
      more pages.
    snoozes: Snoozes matching this list call.
  """
    nextPageToken = _messages.StringField(1)
    snoozes = _messages.MessageField('Snooze', 2, repeated=True)