from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core import properties
class Executions(base.Group):
    """View and manage your Cloud Run jobs executions.

  This set of commands can be used to view and manage your Cloud Run jobs
  executions.
  """
    detailed_help = {'EXAMPLES': '\n          To list your executions for a job, run:\n\n            $ {command} list --job=my-job\n      '}