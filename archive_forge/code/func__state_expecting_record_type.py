import re
from io import BytesIO
from .. import errors
def _state_expecting_record_type(self):
    if len(self._buffer) >= 1:
        record_type = self._buffer[:1]
        self._buffer = self._buffer[1:]
        if record_type == b'B':
            self._state_handler = self._state_expecting_length
        elif record_type == b'E':
            self.finished = True
            self._state_handler = self._state_expecting_nothing
        else:
            raise UnknownRecordTypeError(record_type)