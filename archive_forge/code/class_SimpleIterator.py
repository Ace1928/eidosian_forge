from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
import webob
import glance.api.common
from glance.common import exception
from glance.tests.unit import fixtures as glance_fixtures
class SimpleIterator(object):

    def __init__(self, file_object, chunk_size):
        self.file_object = file_object
        self.chunk_size = chunk_size

    def __iter__(self):

        def read_chunk():
            return self.fobj.read(self.chunk_size)
        chunk = read_chunk()
        while chunk:
            yield chunk
            chunk = read_chunk()
        else:
            return