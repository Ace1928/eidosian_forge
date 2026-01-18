from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
def MakeSetSecurityPolicyRequestTuple(self, security_policy):
    """Makes a call to set the security policy on a backend service."""
    region = getattr(self.ref, 'region', None)
    if region:
        return (self._client.regionBackendServices, 'SetSecurityPolicy', self._messages.ComputeRegionBackendServicesSetSecurityPolicyRequest(securityPolicyReference=self._messages.SecurityPolicyReference(securityPolicy=security_policy), region=region, project=self.ref.project, backendService=self.ref.Name()))
    return (self._client.backendServices, 'SetSecurityPolicy', self._messages.ComputeBackendServicesSetSecurityPolicyRequest(securityPolicyReference=self._messages.SecurityPolicyReference(securityPolicy=security_policy), project=self.ref.project, backendService=self.ref.Name()))