from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListApplicationEnvironmentsResponse(_messages.Message):
    """Message for response to listing ApplicationEnvironments

  Fields:
    applicationEnvironments: Required. The list of ApplicationEnvironment
    nextPageToken: Output only. A token identifying a page of results the
      server should return.
    unreachable: Output only. Locations that could not be reached.
  """
    applicationEnvironments = _messages.MessageField('ApplicationEnvironment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)