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
def _CreatePatchRollout(args, messages):
    """Creates a PatchRollout message from input arguments."""
    if not any([args.rollout_mode, args.rollout_disruption_budget, args.rollout_disruption_budget_percent]):
        return None
    if args.rollout_mode and (not (args.rollout_disruption_budget or args.rollout_disruption_budget_percent)):
        raise exceptions.InvalidArgumentException('rollout-mode', '[rollout-disruption-budget] or [rollout-disruption-budget-percent] must also be specified.')
    if args.rollout_disruption_budget and (not args.rollout_mode):
        raise exceptions.InvalidArgumentException('rollout-disruption-budget', '[rollout-mode] must also be specified.')
    if args.rollout_disruption_budget_percent and (not args.rollout_mode):
        raise exceptions.InvalidArgumentException('rollout-disruption-budget-percent', '[rollout-mode] must also be specified.')
    rollout_modes = messages.PatchRollout.ModeValueValuesEnum
    return messages.PatchRollout(mode=arg_utils.ChoiceToEnum(args.rollout_mode, rollout_modes), disruptionBudget=messages.FixedOrPercent(fixed=int(args.rollout_disruption_budget) if args.rollout_disruption_budget else None, percent=int(args.rollout_disruption_budget_percent) if args.rollout_disruption_budget_percent else None))