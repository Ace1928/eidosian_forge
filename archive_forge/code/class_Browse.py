from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import browser_dispatcher
from googlecloudsdk.command_lib.app import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class Browse(base.Command):
    """Open the specified service(s) in a browser.

  """
    detailed_help = {'EXAMPLES': '          To show the url for the default service in the browser, run:\n\n              $ {command} default\n\n          To show version `v1` of service `myService` in the browser, run:\n\n              $ {command} myService --version="v1"\n\n          To show multiple services side-by-side, run:\n\n              $ {command} default otherService\n\n          To show multiple services side-by-side with a specific version, run:\n\n              $ {command} s1 s2 --version=v1\n          '}

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('\n          table(\n            service:label=SERVICE,\n            url:label=URL\n          )\n    ')
        flags.LAUNCH_BROWSER.AddToParser(parser)
        parser.add_argument('services', nargs='+', help='        The services to open (optionally filtered by the --version flag).')
        parser.add_argument('--version', '-v', help="            If specified, open services with a given version. If not\n            specified, use a version based on the service's traffic split .\n            ")

    def Run(self, args):
        """Launch a browser, or return a table of URLs to print."""
        project = properties.VALUES.core.project.Get(required=True)
        outputs = []
        for service in args.services:
            outputs.append(browser_dispatcher.BrowseApp(project, service, args.version, args.launch_browser))
        if any(outputs):
            if args.launch_browser:
                log.status.Print('Did not detect your browser. Go to these links to view your app:')
            return outputs
        return None