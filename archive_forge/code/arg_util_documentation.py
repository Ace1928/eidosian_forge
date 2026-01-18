from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_file
from googlecloudsdk.api_lib.firebase.test import arg_validate
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
import six
Apply lower-priority arg values from a dictionary to args without values.

  May be used to apply arg default values, or to merge args from another source,
  such as an arg-file. Args which already have a value are never modified by
  this function. Thus, if there are multiple sets of lower-priority args, they
  should be applied in order from highest-to-lowest precedence.

  Args:
    args: the existing argparse.Namespace. All the arguments that were provided
      to the command invocation (i.e. group and command arguments combined),
      plus any arg defaults already applied to the namespace. These args have
      higher priority than the lower_pri_args.
    lower_pri_args: a dict mapping lower-priority arg names to their values.
    issue_cli_warning: (boolean) issue a warning if an arg already has a value
      from the command line and we do not apply the lower-priority arg value
      (used for arg-files where any args specified in the file are lower in
      priority than the CLI args.).
  