from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.vmware import util
def GetPrincipal(self, dns_bind_permission, user=None, service_account=None):
    if user is not None:
        dns_bind_permission.principal = self.messages.Principal(user=user)
    else:
        dns_bind_permission.principal = self.messages.Principal(serviceAccount=service_account)