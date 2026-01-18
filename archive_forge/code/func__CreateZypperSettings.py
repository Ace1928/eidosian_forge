from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _CreateZypperSettings(args, messages):
    """Creates a ZypperSettings message from input arguments."""
    if not any([args.zypper_categories, args.zypper_severities, args.zypper_with_optional, args.zypper_with_update, args.zypper_excludes, args.zypper_exclusive_patches]):
        return None
    return messages.ZypperSettings(categories=args.zypper_categories if args.zypper_categories else [], severities=args.zypper_severities if args.zypper_severities else [], withOptional=args.zypper_with_optional, withUpdate=args.zypper_with_update, excludes=args.zypper_excludes if args.zypper_excludes else [], exclusivePatches=args.zypper_exclusive_patches if args.zypper_exclusive_patches else [])