import pytest
from dummyserver.testcase import (
from urllib3 import HTTPConnectionPool
from urllib3.util import SKIP_HEADER
from urllib3.util.retry import Retry
def _get_header_lines(self, prefix):
    header_block = self.buffer.split(b'\r\n\r\n', 1)[0].lower()
    header_lines = header_block.split(b'\r\n')[1:]
    return [x for x in header_lines if x.startswith(prefix)]