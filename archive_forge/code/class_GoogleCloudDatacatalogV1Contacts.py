from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1Contacts(_messages.Message):
    """Contact people for the entry.

  Fields:
    people: The list of contact people for the entry.
  """
    people = _messages.MessageField('GoogleCloudDatacatalogV1ContactsPerson', 1, repeated=True)