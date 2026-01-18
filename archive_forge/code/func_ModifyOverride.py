from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def ModifyOverride(self, name, overrides, operation_type, update_mask, profile_type='THREAT_PREVENTION', labels=None):
    """Modify the existing threat prevention profile."""
    etag, existing_threat_prevention_profile_object = self.GetSecurityProfileEntities(name)
    updated_threat_prevention_profile_object = self.UpdateThreatPreventionProfile(existing_threat_prevention_profile_object, overrides, update_mask, operation_type)
    if updated_threat_prevention_profile_object == existing_threat_prevention_profile_object:
        update_mask = '*'
    else:
        update_mask = 'threatPreventionProfile'
    security_profile = self.messages.SecurityProfile(name=name, threatPreventionProfile=encoding.DictToMessage(updated_threat_prevention_profile_object, self.messages.ThreatPreventionProfile), etag=etag, type=self._ParseSecurityProfileType(profile_type), labels=labels)
    api_request = self.messages.NetworksecurityOrganizationsLocationsSecurityProfilesPatchRequest(name=name, securityProfile=security_profile, updateMask=update_mask)
    return self._security_profile_client.Patch(api_request)