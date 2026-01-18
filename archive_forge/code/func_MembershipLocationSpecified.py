from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def MembershipLocationSpecified(args, flag_override=''):
    """Returns whether a membership location is specified in args."""
    if args.IsSpecified('location'):
        return True
    if args.IsKnownAndSpecified('membership') and _LOCATION_RE.search(args.membership) is not None:
        return True
    if args.IsKnownAndSpecified('MEMBERSHIP_NAME') and _LOCATION_RE.search(args.MEMBERSHIP_NAME) is not None:
        return True
    if args.IsKnownAndSpecified('memberships') and all([_LOCATION_RE.search(m) is not None for m in args.memberships]):
        return True
    if args.IsKnownAndSpecified(flag_override) and _LOCATION_RE.search(args.GetValue(flag_override)) is not None:
        return True
    return False