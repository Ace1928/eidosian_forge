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
def _ValidatePrePostPatchStepArgs(executable_arg_name, executable_arg, success_codes_arg_name, success_codes_arg):
    """Validates the relation between pre-/post-patch setting flags."""
    if success_codes_arg and (not executable_arg):
        raise exceptions.InvalidArgumentException(success_codes_arg_name, '[{}] must also be specified.'.format(executable_arg_name))