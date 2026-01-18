from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import policies as policies_api
from googlecloudsdk.api_lib.cloudresourcemanager import organizations
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.meta import cache_util as meta_cache_util
from googlecloudsdk.command_lib.util import cache_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ValidateAccessPolicyArg(ref, args, req=None):
    """Add the particular service filter message based on specified args."""
    del ref
    if args.IsSpecified('policy'):
        properties.AccessPolicyValidator(args.policy)
    return req