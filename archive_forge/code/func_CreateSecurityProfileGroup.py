from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def CreateSecurityProfileGroup(self, security_profile_group_name, security_profile_group_id, parent, description, threat_prevention_profile, labels=None):
    """Calls the Create Security Profile Group API."""
    security_profile_group = self.messages.SecurityProfileGroup(name=security_profile_group_name, description=description, threatPreventionProfile=threat_prevention_profile, labels=labels)
    api_request = self.messages.NetworksecurityOrganizationsLocationsSecurityProfileGroupsCreateRequest(parent=parent, securityProfileGroup=security_profile_group, securityProfileGroupId=security_profile_group_id)
    return self._security_profile_group_client.Create(api_request)