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
def AddKonletArgs(parser):
    """Adds Konlet-related args."""
    parser.add_argument('--container-image', help='      Full container image name, which should be pulled onto VM instance,\n      eg. `docker.io/tomcat`.\n      ')
    parser.add_argument('--container-command', help='      Specifies what executable to run when the container starts (overrides\n      default entrypoint), eg. `nc`.\n\n      Default: None (default container entrypoint is used)\n      ')
    parser.add_argument('--container-arg', action='append', help='      Argument to append to container entrypoint or to override container CMD.\n      Each argument must have a separate flag. Arguments are appended in the\n      order of flags. Example:\n\n      Assuming the default entry point of the container (or an entry point\n      overridden with --container-command flag) is a Bourne shell-compatible\n      executable, in order to execute \'ls -l\' command in the container,\n      the user could use:\n\n      `--container-arg="-c" --container-arg="ls -l"`\n\n      Caveat: due to the nature of the argument parsing, it\'s impossible to\n      provide the flag value that starts with a dash (`-`) without the `=` sign\n      (that is, `--container-arg "-c"` will not work correctly).\n\n      Default: None. (no arguments appended)\n      ')
    parser.add_argument('--container-privileged', action='store_true', help='      Specify whether to run container in privileged mode.\n\n      Default: `--no-container-privileged`.\n      ')
    _AddContainerMountHostPathFlag(parser)
    _AddContainerMountTmpfsFlag(parser)
    parser.add_argument('--container-env', type=arg_parsers.ArgDict(), action='append', metavar='KEY=VALUE, ...', help='      Declare environment variables KEY with value VALUE passed to container.\n      Only the last value of KEY is taken when KEY is repeated more than once.\n\n      Values, declared with --container-env flag override those with the same\n      KEY from file, provided in --container-env-file.\n      ')
    parser.add_argument('--container-env-file', help='      Declare environment variables in a file. Values, declared with\n      --container-env flag override those with the same KEY from file.\n\n      File with environment variables in format used by docker (almost).\n      This means:\n      - Lines are in format KEY=VALUE.\n      - Values must contain equality signs.\n      - Variables without values are not supported (this is different from\n        docker format).\n      - If `#` is first non-whitespace character in a line the line is ignored\n        as a comment.\n      - Lines with nothing but whitespace are ignored.\n      ')
    parser.add_argument('--container-stdin', action='store_true', help='      Keep container STDIN open even if not attached.\n\n      Default: `--no-container-stdin`.\n      ')
    parser.add_argument('--container-tty', action='store_true', help='      Allocate a pseudo-TTY for the container.\n\n      Default: `--no-container-tty`.\n      ')
    parser.add_argument('--container-restart-policy', choices=['never', 'on-failure', 'always'], default='always', metavar='POLICY', type=lambda val: val.lower(), help='      Specify whether to restart a container on exit.\n      ')