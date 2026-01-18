from __future__ import absolute_import
import typing
from typing import Any
from typing import Union
from google.cloud.pubsub_v1 import futures
from google.cloud.pubsub_v1.subscriber.exceptions import AcknowledgeStatus
class StreamingPullFuture(futures.Future):
    """Represents a process that asynchronously performs streaming pull and
    schedules messages to be processed.

    This future is resolved when the process is stopped (via :meth:`cancel`) or
    if it encounters an unrecoverable error. Calling `.result()` will cause
    the calling thread to block indefinitely.
    """

    def __init__(self, manager: 'StreamingPullManager'):
        super(StreamingPullFuture, self).__init__()
        self.__manager = manager
        self.__manager.add_close_callback(self._on_close_callback)
        self.__cancelled = False

    def _on_close_callback(self, manager: 'StreamingPullManager', result: Any):
        if self.done():
            return
        if result is None:
            self.set_result(True)
        else:
            self.set_exception(result)

    def cancel(self) -> bool:
        """Stops pulling messages and shutdowns the background thread consuming
        messages.

        The method always returns ``True``, as the shutdown is always initiated.
        However, if the background stream is already being shut down or the shutdown
        has completed, this method is a no-op.

        .. versionchanged:: 2.4.1
           The method does not block anymore, it just triggers the shutdown and returns
           immediately. To block until the background stream is terminated, call
           :meth:`result()` after cancelling the future.

        .. versionchanged:: 2.10.0
           The method always returns ``True`` instead of ``None``.
        """
        self.__cancelled = True
        self.__manager.close()
        return True

    def cancelled(self) -> bool:
        """
        Returns:
            ``True`` if the subscription has been cancelled.
        """
        return self.__cancelled