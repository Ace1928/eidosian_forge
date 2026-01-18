from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsGlobalMembershipsInitializeHubRequest(_messages.Message):
    """A GkehubProjectsLocationsGlobalMembershipsInitializeHubRequest object.

  Fields:
    initializeHubRequest: A InitializeHubRequest resource to be passed as the
      request body.
    project: Required. The Hub to initialize, in the format
      `projects/*/locations/*/memberships/*`.
  """
    initializeHubRequest = _messages.MessageField('InitializeHubRequest', 1)
    project = _messages.StringField(2, required=True)