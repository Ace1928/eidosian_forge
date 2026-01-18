from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.scc.hooks import GetOrganization
def DescribeSourceReqHook(ref, args, req):
    """Generate organization name from organization id."""
    del ref
    req.parent = GetOrganization(args)
    return req