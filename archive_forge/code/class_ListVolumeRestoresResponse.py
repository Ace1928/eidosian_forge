from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVolumeRestoresResponse(_messages.Message):
    """Response message for ListVolumeRestores.

  Fields:
    nextPageToken: A token which may be sent as page_token in a subsequent
      `ListVolumeRestores` call to retrieve the next page of results. If this
      field is omitted or empty, then there are no more results to return.
    volumeRestores: The list of VolumeRestores matching the given criteria.
  """
    nextPageToken = _messages.StringField(1)
    volumeRestores = _messages.MessageField('VolumeRestore', 2, repeated=True)