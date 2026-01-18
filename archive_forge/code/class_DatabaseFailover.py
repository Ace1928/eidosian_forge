from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
class DatabaseFailover(base.Command):
    """Run a database failover operation."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddEnvironmentResourceArg(parser, 'for which to trigger a database failover')
        base.ASYNC_FLAG.AddToParser(parser)

    @staticmethod
    def _ValidateEnvironment(env_obj, release_track):
        messages = api_util.GetMessagesModule(release_track=release_track)
        if env_obj.config.resilienceMode is None or env_obj.config.resilienceMode == messages.EnvironmentConfig.ResilienceModeValueValuesEnum.RESILIENCE_MODE_UNSPECIFIED:
            raise command_util.InvalidUserInputError('Cannot trigger a database failover for environments without enabled high resilience mode.')

    def Run(self, args):
        env_ref = args.CONCEPTS.environment.Parse()
        release_track = self.ReleaseTrack()
        env_obj = environments_api_util.Get(env_ref, release_track=release_track)
        self._ValidateEnvironment(env_obj, release_track)
        operation = environments_api_util.DatabaseFailover(env_ref, release_track=release_track)
        if args.async_:
            return self._AsynchronousExecution(env_ref, operation)
        else:
            return self._SynchronousExecution(env_ref, operation)

    def _AsynchronousExecution(self, env_resource, operation):
        details = 'with operation [{0}]'.format(operation.name)
        log.UpdatedResource(env_resource.RelativeName(), kind='environment', is_async=True, details=details)
        return operation

    def _SynchronousExecution(self, env_resource, operation):
        try:
            operations_api_util.WaitForOperation(operation, 'Waiting for [{}] to be updated with [{}]'.format(env_resource.RelativeName(), operation.name), release_track=self.ReleaseTrack())
        except command_util.Error as e:
            raise command_util.Error('Error triggerering a database failover [{}]: {}'.format(env_resource.RelativeName(), six.text_type(e)))