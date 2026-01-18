from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsTopicsGetPartitionsRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsTopicsGetPartitionsRequest object.

  Fields:
    name: Required. The topic whose partition information to return.
  """
    name = _messages.StringField(1, required=True)