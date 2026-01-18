from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import flags
class Revisions(base.Group):
    """View and manage your Cloud Run Workers revisions.

  This set of commands can be used to view and manage your existing Cloud Run
  Workers revisions.
  """
    detailed_help = {'EXAMPLES': '\n          To list your existing worker revisions, run:\n\n            $ {command} list\n      '}