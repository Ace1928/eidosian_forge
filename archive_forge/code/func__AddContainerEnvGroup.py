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
def _AddContainerEnvGroup(parser):
    """Add flags to update the container environment."""
    env_group = parser.add_argument_group()
    env_group.add_argument('--container-env', type=arg_parsers.ArgDict(), action='append', metavar='KEY=VALUE, ...', help='      Update environment variables `KEY` with value `VALUE` passed to\n      container.\n      - Sets `KEY` to the specified value.\n      - Adds `KEY` = `VALUE`, if `KEY` is not yet declared.\n      - Only the last value of `KEY` is taken when `KEY` is repeated more\n      than once.\n\n      Values, declared with `--container-env` flag override those with the\n      same `KEY` from file, provided in `--container-env-file`.\n      ')
    env_group.add_argument('--container-env-file', help='      Update environment variables from a file.\n      Same update rules as for `--container-env` apply.\n      Values, declared with `--container-env` flag override those with the\n      same `KEY` from file.\n\n      File with environment variables declarations in format used by docker\n      (almost). This means:\n      - Lines are in format KEY=VALUE\n      - Values must contain equality signs.\n      - Variables without values are not supported (this is different from\n      docker format).\n      - If # is first non-whitespace character in a line the line is ignored\n      as a comment.\n      ')
    env_group.add_argument('--remove-container-env', type=arg_parsers.ArgList(), action='append', metavar='KEY', help='      Removes environment variables `KEY` from container declaration Does\n      nothing, if a variable is not present.\n      ')