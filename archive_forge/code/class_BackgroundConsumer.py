import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
class BackgroundConsumer(object):
    """A bi-directional stream consumer that runs in a separate thread.

    This maps the consumption of a stream into a callback-based model. It also
    provides :func:`pause` and :func:`resume` to allow for flow-control.

    Example::

        def should_recover(exc):
            return (
                isinstance(exc, grpc.RpcError) and
                exc.code() == grpc.StatusCode.UNAVAILABLE)

        initial_request = example_pb2.StreamingRpcRequest(
            setting='example')

        rpc = ResumeableBidiRpc(
            stub.StreamingRpc,
            initial_request=initial_request,
            should_recover=should_recover)

        def on_response(response):
            print(response)

        consumer = BackgroundConsumer(rpc, on_response)
        consumer.start()

    Note that error handling *must* be done by using the provided
    ``bidi_rpc``'s ``add_done_callback``. This helper will automatically exit
    whenever the RPC itself exits and will not provide any error details.

    Args:
        bidi_rpc (BidiRpc): The RPC to consume. Should not have been
            ``open()``ed yet.
        on_response (Callable[[protobuf.Message], None]): The callback to
            be called for every response on the stream.
    """

    def __init__(self, bidi_rpc, on_response):
        self._bidi_rpc = bidi_rpc
        self._on_response = on_response
        self._paused = False
        self._wake = threading.Condition()
        self._thread = None
        self._operational_lock = threading.Lock()

    def _on_call_done(self, future):
        self.resume()

    def _thread_main(self, ready):
        try:
            ready.set()
            self._bidi_rpc.add_done_callback(self._on_call_done)
            self._bidi_rpc.open()
            while self._bidi_rpc.is_active:
                with self._wake:
                    while self._paused:
                        _LOGGER.debug('paused, waiting for waking.')
                        self._wake.wait()
                        _LOGGER.debug('woken.')
                _LOGGER.debug('waiting for recv.')
                response = self._bidi_rpc.recv()
                _LOGGER.debug('recved response.')
                self._on_response(response)
        except exceptions.GoogleAPICallError as exc:
            _LOGGER.debug('%s caught error %s and will exit. Generally this is due to the RPC itself being cancelled and the error will be surfaced to the calling code.', _BIDIRECTIONAL_CONSUMER_NAME, exc, exc_info=True)
        except Exception as exc:
            _LOGGER.exception('%s caught unexpected exception %s and will exit.', _BIDIRECTIONAL_CONSUMER_NAME, exc)
        _LOGGER.info('%s exiting', _BIDIRECTIONAL_CONSUMER_NAME)

    def start(self):
        """Start the background thread and begin consuming the thread."""
        with self._operational_lock:
            ready = threading.Event()
            thread = threading.Thread(name=_BIDIRECTIONAL_CONSUMER_NAME, target=self._thread_main, args=(ready,))
            thread.daemon = True
            thread.start()
            ready.wait()
            self._thread = thread
            _LOGGER.debug('Started helper thread %s', thread.name)

    def stop(self):
        """Stop consuming the stream and shutdown the background thread."""
        with self._operational_lock:
            self._bidi_rpc.close()
            if self._thread is not None:
                self.resume()
                self._thread.join(1.0)
                if self._thread.is_alive():
                    _LOGGER.warning('Background thread did not exit.')
            self._thread = None

    @property
    def is_active(self):
        """bool: True if the background thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def pause(self):
        """Pauses the response stream.

        This does *not* pause the request stream.
        """
        with self._wake:
            self._paused = True

    def resume(self):
        """Resumes the response stream."""
        with self._wake:
            self._paused = False
            self._wake.notify_all()

    @property
    def is_paused(self):
        """bool: True if the response stream is paused."""
        return self._paused