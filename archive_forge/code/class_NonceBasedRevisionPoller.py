from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.core import exceptions
class NonceBasedRevisionPoller(waiter.OperationPoller):
    """To poll for exactly one revision with the given nonce to appear."""

    def __init__(self, operations, namespace_ref):
        self._operations = operations
        self._namespace = namespace_ref

    def IsDone(self, revisions):
        return bool(revisions)

    def Poll(self, nonce):
        return self._operations.GetRevisionsByNonce(self._namespace, nonce)

    def GetResult(self, revisions):
        if len(revisions) == 1:
            return revisions[0]
        return None