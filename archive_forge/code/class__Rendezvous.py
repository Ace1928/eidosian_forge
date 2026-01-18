import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
class _Rendezvous(future.Future, face.Call):

    def __init__(self, response_future, response_iterator, call):
        self._future = response_future
        self._iterator = response_iterator
        self._call = call

    def cancel(self):
        return self._call.cancel()

    def cancelled(self):
        return self._future.cancelled()

    def running(self):
        return self._future.running()

    def done(self):
        return self._future.done()

    def result(self, timeout=None):
        try:
            return self._future.result(timeout=timeout)
        except grpc.RpcError as rpc_error_call:
            raise _abortion_error(rpc_error_call)
        except grpc.FutureTimeoutError:
            raise future.TimeoutError()
        except grpc.FutureCancelledError:
            raise future.CancelledError()

    def exception(self, timeout=None):
        try:
            rpc_error_call = self._future.exception(timeout=timeout)
            if rpc_error_call is None:
                return None
            else:
                return _abortion_error(rpc_error_call)
        except grpc.FutureTimeoutError:
            raise future.TimeoutError()
        except grpc.FutureCancelledError:
            raise future.CancelledError()

    def traceback(self, timeout=None):
        try:
            return self._future.traceback(timeout=timeout)
        except grpc.FutureTimeoutError:
            raise future.TimeoutError()
        except grpc.FutureCancelledError:
            raise future.CancelledError()

    def add_done_callback(self, fn):
        self._future.add_done_callback(lambda ignored_callback: fn(self))

    def __iter__(self):
        return self

    def _next(self):
        try:
            return next(self._iterator)
        except grpc.RpcError as rpc_error_call:
            raise _abortion_error(rpc_error_call)

    def __next__(self):
        return self._next()

    def next(self):
        return self._next()

    def is_active(self):
        return self._call.is_active()

    def time_remaining(self):
        return self._call.time_remaining()

    def add_abortion_callback(self, abortion_callback):

        def done_callback():
            if self.code() is not grpc.StatusCode.OK:
                abortion_callback(_abortion(self._call))
        registered = self._call.add_callback(done_callback)
        return None if registered else done_callback()

    def protocol_context(self):
        return _InvocationProtocolContext()

    def initial_metadata(self):
        return _metadata.beta(self._call.initial_metadata())

    def terminal_metadata(self):
        return _metadata.beta(self._call.terminal_metadata())

    def code(self):
        return self._call.code()

    def details(self):
        return self._call.details()