import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
class ZipkinDataBuilder:

    @staticmethod
    def build_span(name, trace_id, span_id, parent_id, annotations, bannotations):
        return ttypes.Span(name=name, trace_id=trace_id, id=span_id, parent_id=parent_id, annotations=annotations, binary_annotations=bannotations)

    @staticmethod
    def build_annotation(value, endpoint=None):
        if isinstance(value, str):
            value = value.encode('utf-8')
        assert isinstance(value, bytes)
        return ttypes.Annotation(time.time() * 1000 * 1000, value, endpoint)

    @staticmethod
    def build_binary_annotation(key, value, endpoint=None):
        annotation_type = ttypes.AnnotationType.STRING
        return ttypes.BinaryAnnotation(key, value, annotation_type, endpoint)

    @staticmethod
    def build_endpoint(ipv4=None, port=None, service_name=None):
        if ipv4 is not None:
            ipv4 = ZipkinDataBuilder._ipv4_to_int(ipv4)
        if service_name is None:
            service_name = ZipkinDataBuilder._get_script_name()
        return ttypes.Endpoint(ipv4=ipv4, port=port, service_name=service_name)

    @staticmethod
    def _ipv4_to_int(ipv4):
        return struct.unpack('!i', socket.inet_aton(ipv4))[0]

    @staticmethod
    def _get_script_name():
        return os.path.basename(sys.argv[0])