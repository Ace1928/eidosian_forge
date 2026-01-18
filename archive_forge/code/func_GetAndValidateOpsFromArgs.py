from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def GetAndValidateOpsFromArgs(parsed_args):
    """Validates and returns labels specific args for update.

  At least one of --update-labels, --clear-labels or --remove-labels must be
  provided. The --clear-labels flag *must* be a declared argument, whether it
  was specified on the command line or not.

  Args:
    parsed_args: The parsed args.

  Returns:
    (update_labels, remove_labels)
    update_labels contains values from --labels and --update-labels flags
    respectively.
    remove_labels contains values from --remove-labels flag

  Raises:
    RequiredArgumentException: if all labels arguments are absent.
    AttributeError: if the --clear-labels flag is absent.
  """
    diff = Diff.FromUpdateArgs(parsed_args)
    if not diff.MayHaveUpdates():
        raise calliope_exceptions.RequiredArgumentException('LABELS', 'At least one of --update-labels, --remove-labels, or --clear-labels must be specified.')
    return diff