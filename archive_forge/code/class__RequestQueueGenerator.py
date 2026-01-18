import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
class _RequestQueueGenerator(object):
    """A helper for sending requests to a gRPC stream from a Queue.

    This generator takes requests off a given queue and yields them to gRPC.

    This helper is useful when you have an indeterminate, indefinite, or
    otherwise open-ended set of requests to send through a request-streaming
    (or bidirectional) RPC.

    The reason this is necessary is because gRPC takes an iterator as the
    request for request-streaming RPCs. gRPC consumes this iterator in another
    thread to allow it to block while generating requests for the stream.
    However, if the generator blocks indefinitely gRPC will not be able to
    clean up the thread as it'll be blocked on `next(iterator)` and not be able
    to check the channel status to stop iterating. This helper mitigates that
    by waiting on the queue with a timeout and checking the RPC state before
    yielding.

    Finally, it allows for retrying without swapping queues because if it does
    pull an item off the queue when the RPC is inactive, it'll immediately put
    it back and then exit. This is necessary because yielding the item in this
    case will cause gRPC to discard it. In practice, this means that the order
    of messages is not guaranteed. If such a thing is necessary it would be
    easy to use a priority queue.

    Example::

        requests = request_queue_generator(q)
        call = stub.StreamingRequest(iter(requests))
        requests.call = call

        for response in call:
            print(response)
            q.put(...)

    Note that it is possible to accomplish this behavior without "spinning"
    (using a queue timeout). One possible way would be to use more threads to
    multiplex the grpc end event with the queue, another possible way is to
    use selectors and a custom event/queue object. Both of these approaches
    are significant from an engineering perspective for small benefit - the
    CPU consumed by spinning is pretty minuscule.

    Args:
        queue (queue_module.Queue): The request queue.
        period (float): The number of seconds to wait for items from the queue
            before checking if the RPC is cancelled. In practice, this
            determines the maximum amount of time the request consumption
            thread will live after the RPC is cancelled.
        initial_request (Union[protobuf.Message,
                Callable[None, protobuf.Message]]): The initial request to
            yield. This is done independently of the request queue to allow fo
            easily restarting streams that require some initial configuration
            request.
    """

    def __init__(self, queue, period=1, initial_request=None):
        self._queue = queue
        self._period = period
        self._initial_request = initial_request
        self.call = None

    def _is_active(self):
        return self.call is None or self.call.is_active()

    def __iter__(self):
        if self._initial_request is not None:
            if callable(self._initial_request):
                yield self._initial_request()
            else:
                yield self._initial_request
        while True:
            try:
                item = self._queue.get(timeout=self._period)
            except queue_module.Empty:
                if not self._is_active():
                    _LOGGER.debug('Empty queue and inactive call, exiting request generator.')
                    return
                else:
                    continue
            if item is None:
                _LOGGER.debug('Cleanly exiting request generator.')
                return
            if not self._is_active():
                self._queue.put(item)
                _LOGGER.debug('Inactive call, replacing item on queue and exiting request generator.')
                return
            yield item