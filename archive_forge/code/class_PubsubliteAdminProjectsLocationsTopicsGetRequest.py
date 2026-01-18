from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsTopicsGetRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsTopicsGetRequest object.

  Fields:
    name: Required. The name of the topic whose configuration to return.
  """
    name = _messages.StringField(1, required=True)