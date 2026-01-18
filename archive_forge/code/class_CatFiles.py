import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class CatFiles(StreamResult):
    """Cat file attachments received to a stream."""

    def __init__(self, byte_stream):
        self.stream = subunit.make_stream_binary(byte_stream)

    def status(self, test_id=None, test_status=None, test_tags=None, runnable=True, file_name=None, file_bytes=None, eof=False, mime_type=None, route_code=None, timestamp=None):
        if file_name is not None:
            self.stream.write(file_bytes)
            self.stream.flush()