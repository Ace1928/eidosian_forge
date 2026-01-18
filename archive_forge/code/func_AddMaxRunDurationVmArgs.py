from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddMaxRunDurationVmArgs(parser, is_update=False):
    """Set arguments for specifing max-run-duration and termination-time flags."""
    max_run_duration_help_text = "      Limits how long this VM instance can run, specified as a duration\n      relative to the VM instance's most-recent start time. Format the duration,\n      ``MAX_RUN_DURATION'', as the number of days, hours, minutes, and seconds\n      followed by d, h, m, and s respectively. For example, specify 30m for a\n      duration of 30 minutes or specify 1d2h3m4s for a duration of 1 day,\n      2 hours, 3 minutes, and 4 seconds. Alternatively, to specify a timestamp,\n      use `--termination-time` instead.\n\n      If neither `--max-run-duration` nor `--termination-time` is specified\n      (default), the VM instance runs until prompted by a user action\n      or system event.\n      If either is specified, the VM instance is scheduled to be automatically\n      terminated using the action specified by `--instance-termination-action`.\n      For `--max-run-duration`, the VM instance is automatically terminated when the VM's\n      current runtime reaches ``MAX_RUN_DURATION''. Note: Anytime the VM instance\n      is stopped or suspended,  `--max-run-duration` and (unless the VM uses\n      `--provisioning-model=SPOT`) `--instance-termination-action` are\n      automatically removed from the VM.\n      "
    termination_time_help_text = "\n      Limits how long this VM instance can run, specified as a time.\n      Format the time, ``TERMINATION_TIME'', as a RFC 3339 timestamp. For more\n      information, see https://tools.ietf.org/html/rfc3339.\n      Alternatively, to specify a duration, use `--max-run-duration` instead.\n\n    If neither `--termination-time` nor `--max-run-duration`\n    is specified (default),\n    the VM instance runs until prompted by a user action or system event.\n    If either is specified, the VM instance is scheduled to be automatically\n    terminated using the action specified by `--instance-termination-action`.\n    For `--termination-time`, the VM instance is automatically terminated at the\n    specified timestamp. Note: Anytime the VM instance is stopped or suspended,\n    `--termination-time` and (unless the VM uses `--provisioning-model=SPOT`)\n    `--instance-termination-action` are automatically removed from the VM.\n    "
    if is_update:
        max_run_duration_group = parser.add_group('Max Run Duration', mutex=True)
        max_run_duration_group.add_argument('--clear-max-run-duration', action='store_true', help='        Removes the max-run-duration field from the scheduling options.\n        ')
        max_run_duration_group.add_argument('--max-run-duration', type=arg_parsers.Duration(), help=max_run_duration_help_text)
        termination_time_group = parser.add_group('Termination Time', mutex=True)
        termination_time_group.add_argument('--clear-termination-time', action='store_true', help='        Removes the termination-time field from the scheduling options.\n        ')
        termination_time_group.add_argument('--termination-time', type=arg_parsers.Datetime.Parse, help=termination_time_help_text)
    else:
        parser.add_argument('--max-run-duration', type=arg_parsers.Duration(), help=max_run_duration_help_text)
        parser.add_argument('--termination-time', type=arg_parsers.Datetime.Parse, help=termination_time_help_text)