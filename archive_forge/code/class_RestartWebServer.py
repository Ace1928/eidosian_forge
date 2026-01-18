from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class RestartWebServer(base.Command):
    """Restart web server for a Cloud Composer environment."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        resource_args.AddEnvironmentResourceArg(parser, 'to restart web server for')
        base.ASYNC_FLAG.AddToParser(parser)

    def Run(self, args):
        env_resource = args.CONCEPTS.environment.Parse()
        operation = environments_api_util.RestartWebServer(env_resource, release_track=self.ReleaseTrack())
        if args.async_:
            return self._AsynchronousExecution(env_resource, operation)
        else:
            return self._SynchronousExecution(env_resource, operation)

    def _AsynchronousExecution(self, env_resource, operation):
        details = 'with operation [{0}]'.format(operation.name)
        log.UpdatedResource(env_resource.RelativeName(), kind='environment', is_async=True, details=details)
        return operation

    def _SynchronousExecution(self, env_resource, operation):
        try:
            operations_api_util.WaitForOperation(operation, 'Waiting for [{}] to be updated with [{}]'.format(env_resource.RelativeName(), operation.name), release_track=self.ReleaseTrack())
        except command_util.Error as e:
            raise command_util.Error('Error restarting web server [{}]: {}'.format(env_resource.RelativeName(), six.text_type(e)))