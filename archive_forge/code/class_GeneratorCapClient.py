from concurrent import futures
import logging
import re
import socket
import typing
import unittest
from tornado.concurrent import (
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
class GeneratorCapClient(BaseCapClient):

    @gen.coroutine
    def capitalize(self, request_data):
        logging.debug('capitalize')
        stream = IOStream(socket.socket())
        logging.debug('connecting')
        yield stream.connect(('127.0.0.1', self.port))
        stream.write(utf8(request_data + '\n'))
        logging.debug('reading')
        data = (yield stream.read_until(b'\n'))
        logging.debug('returning')
        stream.close()
        raise gen.Return(self.process_response(data))