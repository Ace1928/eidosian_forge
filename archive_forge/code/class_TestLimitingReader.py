import io
from oslo_config import fixture as config
from oslotest import base as test_base
import webob
from oslo_middleware import sizelimit
class TestLimitingReader(test_base.BaseTestCase):

    def test_limiting_reader(self):
        BYTES = 1024
        bytes_read = 0
        data = io.StringIO('*' * BYTES)
        for chunk in sizelimit.LimitingReader(data, BYTES):
            bytes_read += len(chunk)
        self.assertEqual(BYTES, bytes_read)
        bytes_read = 0
        data = io.StringIO('*' * BYTES)
        reader = sizelimit.LimitingReader(data, BYTES)
        byte = reader.read(1)
        while len(byte) != 0:
            bytes_read += 1
            byte = reader.read(1)
        self.assertEqual(BYTES, bytes_read)

    def test_read_default_value(self):
        BYTES = 1024
        data_str = '*' * BYTES
        data = io.StringIO(data_str)
        reader = sizelimit.LimitingReader(data, BYTES)
        res = reader.read()
        self.assertEqual(data_str, res)

    def test_limiting_reader_fails(self):
        BYTES = 1024

        def _consume_all_iter():
            bytes_read = 0
            data = io.StringIO('*' * BYTES)
            for chunk in sizelimit.LimitingReader(data, BYTES - 1):
                bytes_read += len(chunk)
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, _consume_all_iter)

        def _consume_all_read():
            bytes_read = 0
            data = io.StringIO('*' * BYTES)
            reader = sizelimit.LimitingReader(data, BYTES - 1)
            byte = reader.read(1)
            while len(byte) != 0:
                bytes_read += 1
                byte = reader.read(1)
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, _consume_all_read)