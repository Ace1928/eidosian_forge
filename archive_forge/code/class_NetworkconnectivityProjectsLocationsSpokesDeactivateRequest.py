from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsSpokesDeactivateRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsSpokesDeactivateRequest object.

  Fields:
    deactivateSpokeRequest: A DeactivateSpokeRequest resource to be passed as
      the request body.
    name: Required. The name of the spoke to deactivate.
  """
    deactivateSpokeRequest = _messages.MessageField('DeactivateSpokeRequest', 1)
    name = _messages.StringField(2, required=True)