from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _SynchronousExecution(self, env_resource, operation):
    try:
        operations_api_util.WaitForOperation(operation, 'Waiting for [{}] to be updated with [{}]'.format(env_resource.RelativeName(), operation.name), release_track=self.ReleaseTrack())
        completed_operation = operations_api_util.GetService(self.ReleaseTrack()).Get(api_util.GetMessagesModule(self.ReleaseTrack()).ComposerProjectsLocationsOperationsGetRequest(name=operation.name))
        log.status.Print('\nIf you want to see the result once more, run:')
        log.status.Print('gcloud composer operations describe ' + operation.name + '\n')
        log.status.Print('If you want to see history of all operations to be able to display results of previous check-upgrade runs, run:')
        log.status.Print('gcloud composer operations list\n')
        log.status.Print('Response: ')
        return completed_operation.response
    except command_util.Error as e:
        raise command_util.Error('Failed to save the snapshot of the environment [{}]: {}'.format(env_resource.RelativeName(), six.text_type(e)))