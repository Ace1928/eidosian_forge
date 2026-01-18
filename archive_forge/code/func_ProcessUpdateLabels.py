from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def ProcessUpdateLabels(args, labels_name, labels_cls, orig_labels):
    """Returns the result of applying the diff constructed from args.

  This API doesn't conform to the standard patch semantics, and instead does
  a replace operation on update. Therefore, if there are no updates to do,
  then the original labels must be returned as writing None into the labels
  field would replace it.

  Args:
    args: argparse.Namespace, the parsed arguments with update_labels,
      remove_labels, and clear_labels
    labels_name: str, the name for the labels flag.
    labels_cls: type, the LabelsValue class for the new labels.
    orig_labels: message, the original LabelsValue value to be updated.

  Returns:
    LabelsValue: The updated labels of type labels_cls.

  Raises:
    ValueError: if the update does not change the labels.
  """
    labels_diff = labels_util.Diff(additions=getattr(args, 'update_' + labels_name), subtractions=getattr(args, 'remove_' + labels_name), clear=getattr(args, 'clear_' + labels_name))
    if not labels_diff.MayHaveUpdates():
        return None
    return labels_diff.Apply(labels_cls, orig_labels).GetOrNone()