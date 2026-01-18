import pytest
from dummyserver.testcase import (
from urllib3 import HTTPConnectionPool
from urllib3.util import SKIP_HEADER
from urllib3.util.retry import Retry
def _test_body(self, data):
    self.start_chunked_handler()
    with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
        pool.urlopen('GET', '/', data, chunked=True)
        header, body = self.buffer.split(b'\r\n\r\n', 1)
        assert b'Transfer-Encoding: chunked' in header.split(b'\r\n')
        if data:
            bdata = data if isinstance(data, bytes) else data.encode('utf-8')
            assert b'\r\n' + bdata + b'\r\n' in body
            assert body.endswith(b'\r\n0\r\n\r\n')
            len_str = body.split(b'\r\n', 1)[0]
            stated_len = int(len_str, 16)
            assert stated_len == len(bdata)
        else:
            assert body == b'0\r\n\r\n'