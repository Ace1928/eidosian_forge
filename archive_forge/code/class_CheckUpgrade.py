from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class CheckUpgrade(base.Command):
    """Check that upgrading a Cloud Composer environment does not result in PyPI module conflicts."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddEnvironmentResourceArg(parser, 'to check upgrade for')
        base.ASYNC_FLAG.AddToParser(parser)
        flags.AddEnvUpgradeFlagsToGroup(parser)

    def Run(self, args):
        env_resource = args.CONCEPTS.environment.Parse()
        env_details = environments_api_util.Get(env_resource, self.ReleaseTrack())
        if (args.airflow_version or args.image_version) and image_versions_command_util.IsDefaultImageVersion(args.image_version):
            message = image_versions_command_util.BuildDefaultComposerVersionWarning(args.image_version, args.airflow_version)
            log.warning(message)
        if args.airflow_version:
            args.image_version = image_versions_command_util.ImageVersionFromAirflowVersion(args.airflow_version, env_details.config.softwareConfig.imageVersion)
        if args.image_version:
            upgrade_validation = image_versions_command_util.IsValidImageVersionUpgrade(env_details.config.softwareConfig.imageVersion, args.image_version)
            if not upgrade_validation.upgrade_valid:
                raise command_util.InvalidUserInputError(upgrade_validation.error)
        operation = environments_api_util.CheckUpgrade(env_resource, args.image_version, release_track=self.ReleaseTrack())
        if args.async_:
            return self._AsynchronousExecution(env_resource, operation, args.image_version)
        else:
            return self._SynchronousExecution(env_resource, operation, args.image_version)

    def _AsynchronousExecution(self, env_resource, operation, image_version):
        details = 'to image {0} with operation [{1}]'.format(image_version, operation.name)
        log._PrintResourceChange('check', env_resource.RelativeName(), kind='environment', is_async=True, details=details, failed=None)
        log.Print('If you want to see the result, run:')
        log.Print('gcloud composer operations describe ' + operation.name)

    def _SynchronousExecution(self, env_resource, operation, image_version):
        try:
            operations_api_util.WaitForOperation(operation, 'Waiting for [{}] to be checked for PyPI package conflicts when upgrading to {}. Operation [{}]'.format(env_resource.RelativeName(), image_version, operation.name), release_track=self.ReleaseTrack())
            completed_operation = operations_api_util.GetService(self.ReleaseTrack()).Get(api_util.GetMessagesModule(self.ReleaseTrack()).ComposerProjectsLocationsOperationsGetRequest(name=operation.name))
            log.Print('\nIf you want to see the result once more, run:')
            log.Print('gcloud composer operations describe ' + operation.name + '\n')
            log.Print('If you want to see history of all operations to be able to display results of previous check-upgrade runs, run:')
            log.Print('gcloud composer operations list\n')
            log.Print('Response: ')
            return completed_operation.response
        except command_util.Error as e:
            raise command_util.Error('Error while checking for PyPI package conflicts [{}]: {}'.format(env_resource.RelativeName(), six.text_type(e)))