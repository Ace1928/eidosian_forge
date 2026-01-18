from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.orgpolicy import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages
def DeletePolicy(self, name: str, etag=None) -> orgpolicy_v2_messages.GoogleProtobufEmpty:
    if name.startswith('organizations/'):
        request = self.messages.OrgpolicyOrganizationsPoliciesDeleteRequest(name=name, etag=etag)
        return self.client.organizations_policies.Delete(request)
    elif name.startswith('folders/'):
        request = self.messages.OrgpolicyFoldersPoliciesDeleteRequest(name=name, etag=etag)
        return self.client.folders_policies.Delete(request)
    else:
        request = self.messages.OrgpolicyProjectsPoliciesDeleteRequest(name=name, etag=etag)
        return self.client.projects_policies.Delete(request)