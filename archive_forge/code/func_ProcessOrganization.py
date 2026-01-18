from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ProcessOrganization(ref, args, req):
    """Hook to process organization input."""
    del ref, args
    if req.parent is not None:
        return req
    org = properties.VALUES.access_context_manager.organization.Get()
    if org is None:
        raise calliope_exceptions.RequiredArgumentException('--organization', 'The attribute can be set in the following ways: \n' + '- provide the argument `--organization` on the command line \n' + '- set the property `access_context_manager/organization`')
    org_ref = resources.REGISTRY.Parse(org, collection='accesscontextmanager.organizations')
    req.parent = org_ref.RelativeName()
    return req