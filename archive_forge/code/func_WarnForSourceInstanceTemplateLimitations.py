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
def WarnForSourceInstanceTemplateLimitations(args):
    """Warn if --source-instance-template is mixed with unsupported flags.

  Args:
    args: Argument namespace
  """
    allowed_flags = ['--project', '--zone', '--region', '--source-instance-template', 'INSTANCE_NAMES:1', '--machine-type', '--custom-cpu', '--custom-memory', '--labels']
    if args.IsSpecified('source_instance_template'):
        specified_args = args.GetSpecifiedArgNames()
        for flag in allowed_flags:
            if flag in specified_args:
                specified_args.remove(flag)
        if specified_args:
            log.status.write('When a source instance template is used, additional parameters other than --machine-type and --labels will be ignored but provided by the source instance template\n')