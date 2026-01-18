from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.sessions import (
from googlecloudsdk.command_lib.dataproc.shared_messages import (
from googlecloudsdk.command_lib.dataproc.shared_messages import (
from googlecloudsdk.command_lib.util.args import labels_util
def AddArguments(parser):
    """Adds arguments related to Session message.

  Add Session arguments to the given parser. Session specific arguments are not
  handled, and need to be set during factory instantiation.

  Args:
    parser: A argument parser.
  """
    parser.add_argument('--session_template', help='The session template to use for creating the session.')
    labels_util.AddCreateLabelsFlags(parser)
    _AddDependency(parser)