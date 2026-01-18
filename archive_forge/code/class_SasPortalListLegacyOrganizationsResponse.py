from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalListLegacyOrganizationsResponse(_messages.Message):
    """Response for [ListLegacyOrganizations].
  [spectrum.sas.portal.v1alpha1.Provisioning.ListLegacyOrganizations].

  Fields:
    organizations: Optional. Legacy SAS organizations.
  """
    organizations = _messages.MessageField('SasPortalOrganization', 1, repeated=True)