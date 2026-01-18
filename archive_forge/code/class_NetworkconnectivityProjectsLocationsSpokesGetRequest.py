from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsSpokesGetRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsSpokesGetRequest object.

  Fields:
    name: Required. The name of the spoke resource.
  """
    name = _messages.StringField(1, required=True)