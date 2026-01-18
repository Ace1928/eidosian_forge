from binascii import crc32
from struct import unpack
from botocore.exceptions import EventStreamError
def _create_raw_event_generator(self):
    event_stream_buffer = EventStreamBuffer()
    for chunk in self._raw_stream.stream():
        event_stream_buffer.add_data(chunk)
        yield from event_stream_buffer