from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.metastore import util as api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
class _OperationPollerWithError(waiter.CloudOperationPollerNoResources):
    """Class for polling Metastore longrunning Operations and print errors."""

    def __init__(self, release_track=base.ReleaseTrack.GA):
        super(_OperationPollerWithError, self).__init__(GetOperation(release_track=release_track), lambda x: x)

    def IsDone(self, operation):
        if not operation.done:
            return False
        if operation.error:
            if operation.error.code:
                log.status.Print('Status Code:', operation.error.code)
            if operation.error.message:
                log.status.Print('Error message:', operation.error.message)
            if operation.error.details:
                for message in operation.error.details[0].additionalProperties:
                    if message.key == 'details':
                        log.status.Print('Error details:', message.value.object_value.properties[0].value.string_value)
            raise api_util.OperationError(operation.name, '')
        return True