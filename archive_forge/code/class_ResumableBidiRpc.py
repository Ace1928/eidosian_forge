import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
class ResumableBidiRpc(BidiRpc):
    """A :class:`BidiRpc` that can automatically resume the stream on errors.

    It uses the ``should_recover`` arg to determine if it should re-establish
    the stream on error.

    Example::

        def should_recover(exc):
            return (
                isinstance(exc, grpc.RpcError) and
                exc.code() == grpc.StatusCode.UNAVAILABLE)

        initial_request = example_pb2.StreamingRpcRequest(
            setting='example')

        metadata = [('header_name', 'value')]

        rpc = ResumableBidiRpc(
            stub.StreamingRpc,
            should_recover=should_recover,
            initial_request=initial_request,
            metadata=metadata
        )

        rpc.open()

        while rpc.is_active():
            print(rpc.recv())
            rpc.send(example_pb2.StreamingRpcRequest(
                data='example'))

    Args:
        start_rpc (grpc.StreamStreamMultiCallable): The gRPC method used to
            start the RPC.
        initial_request (Union[protobuf.Message,
                Callable[None, protobuf.Message]]): The initial request to
            yield. This is useful if an initial request is needed to start the
            stream.
        should_recover (Callable[[Exception], bool]): A function that returns
            True if the stream should be recovered. This will be called
            whenever an error is encountered on the stream.
        should_terminate (Callable[[Exception], bool]): A function that returns
            True if the stream should be terminated. This will be called
            whenever an error is encountered on the stream.
        metadata Sequence[Tuple(str, str)]: RPC metadata to include in
            the request.
        throttle_reopen (bool): If ``True``, throttling will be applied to
            stream reopen calls. Defaults to ``False``.
    """

    def __init__(self, start_rpc, should_recover, should_terminate=_never_terminate, initial_request=None, metadata=None, throttle_reopen=False):
        super(ResumableBidiRpc, self).__init__(start_rpc, initial_request, metadata)
        self._should_recover = should_recover
        self._should_terminate = should_terminate
        self._operational_lock = threading.RLock()
        self._finalized = False
        self._finalize_lock = threading.Lock()
        if throttle_reopen:
            self._reopen_throttle = _Throttle(access_limit=5, time_window=datetime.timedelta(seconds=10))
        else:
            self._reopen_throttle = None

    def _finalize(self, result):
        with self._finalize_lock:
            if self._finalized:
                return
            for callback in self._callbacks:
                callback(result)
            self._finalized = True

    def _on_call_done(self, future):
        with self._operational_lock:
            if self._should_terminate(future):
                self._finalize(future)
            elif not self._should_recover(future):
                self._finalize(future)
            else:
                _LOGGER.debug('Re-opening stream from gRPC callback.')
                self._reopen()

    def _reopen(self):
        with self._operational_lock:
            if self.call is not None and self.call.is_active():
                _LOGGER.debug('Stream was already re-established.')
                return
            self.call = None
            self._request_generator = None
            try:
                if self._reopen_throttle:
                    with self._reopen_throttle:
                        self.open()
                else:
                    self.open()
            except Exception as exc:
                _LOGGER.debug('Failed to re-open stream due to %s', exc)
                self._finalize(exc)
                raise
            _LOGGER.info('Re-established stream')

    def _recoverable(self, method, *args, **kwargs):
        """Wraps a method to recover the stream and retry on error.

        If a retryable error occurs while making the call, then the stream will
        be re-opened and the method will be retried. This happens indefinitely
        so long as the error is a retryable one. If an error occurs while
        re-opening the stream, then this method will raise immediately and
        trigger finalization of this object.

        Args:
            method (Callable[..., Any]): The method to call.
            args: The args to pass to the method.
            kwargs: The kwargs to pass to the method.
        """
        while True:
            try:
                return method(*args, **kwargs)
            except Exception as exc:
                with self._operational_lock:
                    _LOGGER.debug('Call to retryable %r caused %s.', method, exc)
                    if self._should_terminate(exc):
                        self.close()
                        _LOGGER.debug('Terminating %r due to %s.', method, exc)
                        self._finalize(exc)
                        break
                    if not self._should_recover(exc):
                        self.close()
                        _LOGGER.debug('Not retrying %r due to %s.', method, exc)
                        self._finalize(exc)
                        raise exc
                    _LOGGER.debug('Re-opening stream from retryable %r.', method)
                    self._reopen()

    def _send(self, request):
        with self._operational_lock:
            call = self.call
        if call is None:
            raise ValueError('Can not send() on an RPC that has never been open()ed.')
        if call.is_active():
            self._request_queue.put(request)
            pass
        else:
            next(call)

    def send(self, request):
        return self._recoverable(self._send, request)

    def _recv(self):
        with self._operational_lock:
            call = self.call
        if call is None:
            raise ValueError('Can not recv() on an RPC that has never been open()ed.')
        return next(call)

    def recv(self):
        return self._recoverable(self._recv)

    def close(self):
        self._finalize(None)
        super(ResumableBidiRpc, self).close()

    @property
    def is_active(self):
        """bool: True if this stream is currently open and active."""
        with self._operational_lock:
            return self.call is not None and (not self._finalized)