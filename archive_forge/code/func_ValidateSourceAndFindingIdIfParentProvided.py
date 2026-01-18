from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def ValidateSourceAndFindingIdIfParentProvided(args):
    """Validates that source and finding id are provided if parent is provided."""
    if args.source is None:
        raise errors.InvalidSCCInputError('--source flag must be provided.')
    if '/' in args.finding:
        raise errors.InvalidSCCInputError('Finding id must be provided, instead of the full resource name.')