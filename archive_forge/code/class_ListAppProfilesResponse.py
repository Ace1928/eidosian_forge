from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAppProfilesResponse(_messages.Message):
    """Response message for BigtableInstanceAdmin.ListAppProfiles.

  Fields:
    appProfiles: The list of requested app profiles.
    failedLocations: Locations from which AppProfile information could not be
      retrieved, due to an outage or some other transient condition.
      AppProfiles from these locations may be missing from `app_profiles`.
      Values are of the form `projects//locations/`
    nextPageToken: Set if not all app profiles could be returned in a single
      response. Pass this value to `page_token` in another request to get the
      next page of results.
  """
    appProfiles = _messages.MessageField('AppProfile', 1, repeated=True)
    failedLocations = _messages.StringField(2, repeated=True)
    nextPageToken = _messages.StringField(3)