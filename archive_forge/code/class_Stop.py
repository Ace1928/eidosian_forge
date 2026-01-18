from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
import six
class Stop(base.Command):
    """Stop serving specified versions.

  This command stops serving the specified versions. It may only be used if the
  scaling module for your service has been set to manual.
  """
    detailed_help = {'EXAMPLES': '          To stop a specific version across all services, run:\n\n            $ {command} v1\n\n          To stop multiple named versions across all services, run:\n\n            $ {command} v1 v2\n\n          To stop a single version on a single service, run:\n\n            $ {command} --service=servicename v1\n\n          To stop multiple versions in a single service, run:\n\n            $ {command} --service=servicename v1 v2\n\n          Note that that last example may be more simply written using the\n          `services stop` command (see its documentation for details).\n          '}

    @staticmethod
    def Args(parser):
        parser.add_argument('versions', nargs='+', help='The versions to stop (optionally filtered by the --service flag).')
        parser.add_argument('--service', '-s', help='If specified, only stop versions from the given service.')

    def Run(self, args):
        api_client = appengine_api_client.GetApiClientForTrack(self.ReleaseTrack())
        services = api_client.ListServices()
        versions = version_util.GetMatchingVersions(api_client.ListVersions(services), args.versions, args.service)
        if versions:
            fmt = 'list[title="Stopping the following versions:"]'
            resource_printer.Print(versions, fmt, out=log.status)
            console_io.PromptContinue(cancel_on_no=True)
        else:
            log.warning('No matching versions found.')
        errors = []
        for version in sorted(versions, key=str):
            try:
                with progress_tracker.ProgressTracker('Stopping [{0}]'.format(version)):
                    operations_util.CallAndCollectOpErrors(api_client.StopVersion, version.service, version.id, block=True)
            except operations_util.MiscOperationError as err:
                errors.append(six.text_type(err))
        if errors:
            raise VersionsStopError('\n\n'.join(errors))