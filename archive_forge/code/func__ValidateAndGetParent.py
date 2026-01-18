from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def _ValidateAndGetParent(args):
    """Validates parent."""
    if args.organization is not None:
        return _NormalizeOrganization(args.organization)
    if args.folder is not None:
        return _NormalizeFolder(args.folder)
    if args.project is not None:
        return _NormalizeProject(args.project)
    return None