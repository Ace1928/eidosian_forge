from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import browser_dispatcher
from googlecloudsdk.core import properties
from six.moves import urllib
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA)
class OpenConsole(base.Command):
    """Open the App Engine dashboard, or log viewer, in a web browser.

  """
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          Open the App Engine dashboard for the default service:\n\n              $ {command}\n\n          Open the service specific dashboard view:\n\n              $ {command} --service="myService"\n\n          Open the version specific dashboard view:\n\n              $ {command} --service="myService" --version="v1"\n\n          Open the log viewer for the default service:\n\n              $ {command} --logs\n          '}

    @staticmethod
    def Args(parser):
        parser.add_argument('--service', '-s', help='The service to consider. If not specified, use the default service.')
        parser.add_argument('--version', '-v', help='The version to consider. If not specified, all versions for the given service are considered.')
        parser.add_argument('--logs', '-l', action='store_true', default=False, help='Open the log viewer instead of the App Engine dashboard.')

    def Run(self, args):
        project = properties.VALUES.core.project.Get(required=True)
        url = _CreateDevConsoleURL(project, args.service, args.version, args.logs)
        browser_dispatcher.OpenURL(url)