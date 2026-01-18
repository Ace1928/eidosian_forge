from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def DeleteOverride(self, name, overrides, update_mask, profile_type='THREAT_PREVENTION', labels=None):
    """Delete the existing threat prevention profile override."""
    etag, existing_threat_prevention_profile_object = self.GetSecurityProfileEntities(name)
    if update_mask in existing_threat_prevention_profile_object:
        update_field = ''
        if update_mask == 'severityOverrides':
            update_field = 'severity'
        elif update_mask == 'threatOverrides':
            update_field = 'threatId'
        for specified_override in overrides:
            for i in range(0, len(existing_threat_prevention_profile_object.get(update_mask))):
                if existing_threat_prevention_profile_object.get(update_mask)[i].get(update_field) == specified_override:
                    existing_threat_prevention_profile_object.get(update_mask).pop(i)
                    break
    security_profile = self.messages.SecurityProfile(name=name, threatPreventionProfile=encoding.DictToMessage(existing_threat_prevention_profile_object, self.messages.ThreatPreventionProfile), etag=etag, type=self._ParseSecurityProfileType(profile_type), labels=labels)
    api_request = self.messages.NetworksecurityOrganizationsLocationsSecurityProfilesPatchRequest(name=name, securityProfile=security_profile, updateMask='threatPreventionProfile')
    return self._security_profile_client.Patch(api_request)