from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.components import util
class Reinstall(base.SilentCommand):
    """Reinstall the Google Cloud CLI with the same components you have now.

  If your Google Cloud CLI installation becomes corrupt, this command attempts
  to fix it by downloading the latest version of the Google Cloud CLI and
  reinstalling it. This will replace your existing installation with a fresh
  one.  The command is the equivalent of deleting your current installation,
  downloading a fresh copy of the gcloud CLI, and installing in the same
  location.

  ## EXAMPLES
  To reinstall all components you have installed, run:

    $ {command}
  """

    @staticmethod
    def Args(parser):
        pass

    def Run(self, args):
        """Runs the list command."""
        update_manager = util.GetUpdateManager(args)
        update_manager.Reinstall()