import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
class TraceData:
    END_ANNOTATION = SERVER_SEND

    def __init__(self, name, trace_id, span_id, parent_id, sampled, endpoint):
        """
        :param name: RPC name (String)
        :param trace_id: int
        :param span_id: int
        :param parent_id: int or None
        :param sampled: lets the downstream servers know
                    if I should record trace data for the request (bool)
        :param endpoint: zipkin._thrift.zipkinCore.ttypes.EndPoint
        """
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.sampled = sampled
        self.endpoint = endpoint
        self.annotations = []
        self.bannotations = []
        self._done = False

    def add_annotation(self, annotation):
        if annotation.host is None:
            annotation.host = self.endpoint
        if not self._done:
            self.annotations.append(annotation)
            if annotation.value == self.END_ANNOTATION:
                self.flush()

    def add_binary_annotation(self, bannotation):
        if bannotation.host is None:
            bannotation.host = self.endpoint
        if not self._done:
            self.bannotations.append(bannotation)

    def flush(self):
        span = ZipkinDataBuilder.build_span(name=self.name, trace_id=self.trace_id, span_id=self.span_id, parent_id=self.parent_id, annotations=self.annotations, bannotations=self.bannotations)
        client.send_to_collector(span)
        self.annotations = []
        self.bannotations = []
        self._done = True