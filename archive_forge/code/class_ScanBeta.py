from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.ondemandscanning import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import ondemandscanning_util as ods_util
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import platforms
import six
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ScanBeta(base.Command):
    """Perform a vulnerability scan on a container image.

  You can scan a container image in a Google Cloud registry (Artifact Registry
  or Container Registry), or a local container image.

  Reference an image by tag or digest using any of the formats:

    Artifact Registry:
      LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE[:tag]
      LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE@sha256:digest

    Container Registry:
      [LOCATION.]gcr.io/PROJECT-ID/REPOSITORY-ID/IMAGE[:tag]
      [LOCATION.]gcr.io/PROJECT-ID/REPOSITORY-ID/IMAGE@sha256:digest

    Local:
      IMAGE[:tag]
  """
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '    Start a scan of a container image stored in Artifact Registry:\n\n        $ {command} us-west1-docker.pkg.dev/my-project/my-repository/busy-box@sha256:abcxyz --remote\n\n    Start a scan of a container image stored in the Container Registry, and perform the analysis in Europe:\n\n        $ {command} eu.gcr.io/my-project/my-repository/my-image:latest --remote --location=europe\n\n    Start a scan of a container image stored locally, and perform the analysis in Asia:\n\n        $ {command} ubuntu:latest --location=asia\n    '}

    @staticmethod
    def Args(parser):
        flags.GetResourceURIArg().AddToParser(parser)
        flags.GetRemoteFlag().AddToParser(parser)
        flags.GetOnDemandScanningFakeExtractionFlag().AddToParser(parser)
        flags.GetOnDemandScanningLocationFlag().AddToParser(parser)
        flags.GetAdditionalPackageTypesFlag().AddToParser(parser)
        flags.GetExperimentalPackageTypesFlag().AddToParser(parser)
        flags.GetVerboseErrorsFlag().AddToParser(parser)
        base.ASYNC_FLAG.AddToParser(parser)

    def Run(self, args):
        """Runs local extraction then calls ODS with the results.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      AnalyzePackages operation.

    Raises:
      UnsupportedOS: when the command is run on a Windows machine.
    """
        if platforms.OperatingSystem.IsWindows():
            raise ods_util.UnsupportedOS('On-Demand Scanning is not supported on Windows')
        try:
            update_manager.UpdateManager.EnsureInstalledAndRestart(['local-extract'])
        except update_manager.MissingRequiredComponentsError:
            raise
        except local_state.InvalidSDKRootError:
            pass
        cmd = Command()
        stages = [progress_tracker.Stage(EXTRACT_MESSAGE.format('remote' if args.remote else 'local'), key='extract'), progress_tracker.Stage(RPC_MESSAGE, key='rpc')]
        if not args.async_:
            stages += [progress_tracker.Stage(POLL_MESSAGE, key='poll')]
        messages = self.GetMessages()
        with progress_tracker.StagedProgressTracker(SCAN_MESSAGE, stages=stages) as tracker:
            tracker.StartStage('extract')
            operation_result = cmd(resource_uri=args.RESOURCE_URI, remote=args.remote, fake_extraction=args.fake_extraction, additional_package_types=args.additional_package_types, experimental_package_types=args.experimental_package_types, verbose_errors=args.verbose_errors)
            if operation_result.exit_code:
                extraction_error = None
                if operation_result.stderr:
                    extraction_error = '\n'.join([line for line in operation_result.stderr.splitlines() if line.startswith('Extraction failed')])
                if not extraction_error:
                    if operation_result.exit_code < 0:
                        extraction_error = EXTRACTION_KILLED_ERROR_TEMPLATE.format(exit_code=operation_result.exit_code)
                    else:
                        extraction_error = UNKNOWN_EXTRACTION_ERROR_TEMPLATE.format(exit_code=operation_result.exit_code)
                tracker.FailStage('extract', ods_util.ExtractionFailedError(extraction_error))
                return
            pkgs = []
            for pkg in json.loads(operation_result.stdout):
                pkg_data = messages.PackageData(package=pkg['package'], version=pkg['version'], cpeUri=pkg['cpe_uri'])
                if 'package_type' in pkg:
                    pkg_data.packageType = arg_utils.ChoiceToEnum(pkg['package_type'], messages.PackageData.PackageTypeValueValuesEnum)
                if 'hash_digest' in pkg:
                    pkg_data.hashDigest = pkg['hash_digest']
                pkgs += [pkg_data]
            tracker.CompleteStage('extract')
            tracker.StartStage('rpc')
            op = self.AnalyzePackages(args, pkgs)
            tracker.CompleteStage('rpc')
            response = None
            if not args.async_:
                tracker.StartStage('poll')
                tracker.UpdateStage('poll', '[{}]'.format(op.name))
                response = self.WaitForOperation(op)
                tracker.CompleteStage('poll')
        if args.async_:
            log.status.Print('Check operation [{}] for status.'.format(op.name))
            return op
        return response

    def AnalyzePackages(self, args, pkgs):
        return api_util.AnalyzePackagesBeta(properties.VALUES.core.project.Get(required=True), args.location, args.RESOURCE_URI, pkgs)

    def GetMessages(self):
        return api_util.GetMessages('v1beta1')

    def WaitForOperation(self, op):
        return ods_util.WaitForOperation(op, 'v1beta1')