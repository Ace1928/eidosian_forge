from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core.console import progress_tracker
class DeletionPoller(waiter.OperationPoller):
    """Polls for deletion of a resource."""

    def __init__(self, getter):
        """Supply getter as the resource getter."""
        self._getter = getter
        self._ret = None

    def IsDone(self, obj):
        return obj is None or obj.conditions.IsFailed()

    def Poll(self, ref):
        try:
            self._ret = self._getter(ref)
        except api_exceptions.HttpNotFoundError:
            self._ret = None
        return self._ret

    def GetMessage(self):
        if self._ret and self._ret.conditions and (not self._ret.conditions.IsReady()):
            return self._ret.conditions.DescriptiveMessage() or ''
        return ''

    def GetResult(self, obj):
        return obj