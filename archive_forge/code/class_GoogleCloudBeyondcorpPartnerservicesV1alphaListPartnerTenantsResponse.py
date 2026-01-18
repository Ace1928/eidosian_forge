from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaListPartnerTenantsResponse(_messages.Message):
    """Message for response to listing PartnerTenants.

  Fields:
    nextPageToken: A token to retrieve the next page of results, or empty if
      there are no more results in the list.
    partnerTenants: The list of PartnerTenant objects.
  """
    nextPageToken = _messages.StringField(1)
    partnerTenants = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaPartnerTenant', 2, repeated=True)