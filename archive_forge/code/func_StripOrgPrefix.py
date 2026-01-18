from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def StripOrgPrefix(org_id):
    prefix = 'organizations/'
    if org_id.startswith(prefix):
        return org_id[len(prefix):]
    else:
        return org_id