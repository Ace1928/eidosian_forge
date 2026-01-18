import collections
import threading
import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import abandonment
from grpc.framework.foundation import logging_pool
from grpc.framework.foundation import stream
from grpc.framework.interfaces.face import face
class _FaceServicerContext(face.ServicerContext):

    def __init__(self, servicer_context):
        self._servicer_context = servicer_context

    def is_active(self):
        return self._servicer_context.is_active()

    def time_remaining(self):
        return self._servicer_context.time_remaining()

    def add_abortion_callback(self, abortion_callback):
        raise NotImplementedError('add_abortion_callback no longer supported server-side!')

    def cancel(self):
        self._servicer_context.cancel()

    def protocol_context(self):
        return _ServerProtocolContext(self._servicer_context)

    def invocation_metadata(self):
        return _metadata.beta(self._servicer_context.invocation_metadata())

    def initial_metadata(self, initial_metadata):
        self._servicer_context.send_initial_metadata(_metadata.unbeta(initial_metadata))

    def terminal_metadata(self, terminal_metadata):
        self._servicer_context.set_terminal_metadata(_metadata.unbeta(terminal_metadata))

    def code(self, code):
        self._servicer_context.set_code(code)

    def details(self, details):
        self._servicer_context.set_details(details)