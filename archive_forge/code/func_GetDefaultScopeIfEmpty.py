from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import properties
def GetDefaultScopeIfEmpty(args):
    """Return the request scope and fall back to core project if not specified."""
    if args.IsSpecified('scope'):
        VerifyScopeForSearch(args.scope)
        return args.scope
    else:
        return 'projects/{0}'.format(properties.VALUES.core.project.GetOrFail())