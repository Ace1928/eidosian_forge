from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsGetRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsGetRequest object.

  Fields:
    name: Resource name for the location.
  """
    name = _messages.StringField(1, required=True)