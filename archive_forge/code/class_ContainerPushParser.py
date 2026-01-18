import re
from io import BytesIO
from .. import errors
class ContainerPushParser:
    """A "push" parser for container format 1.

    It accepts bytes via the ``accept_bytes`` method, and parses them into
    records which can be retrieved via the ``read_pending_records`` method.
    """

    def __init__(self):
        self._buffer = b''
        self._state_handler = self._state_expecting_format_line
        self._parsed_records = []
        self._reset_current_record()
        self.finished = False

    def _reset_current_record(self):
        self._current_record_length = None
        self._current_record_names = []

    def accept_bytes(self, bytes):
        self._buffer += bytes
        last_buffer_length = None
        cur_buffer_length = len(self._buffer)
        last_state_handler = None
        while cur_buffer_length != last_buffer_length or last_state_handler != self._state_handler:
            last_buffer_length = cur_buffer_length
            last_state_handler = self._state_handler
            self._state_handler()
            cur_buffer_length = len(self._buffer)

    def read_pending_records(self, max=None):
        if max:
            records = self._parsed_records[:max]
            del self._parsed_records[:max]
            return records
        else:
            records = self._parsed_records
            self._parsed_records = []
            return records

    def _consume_line(self):
        """Take a line out of the buffer, and return the line.

        If a newline byte is not found in the buffer, the buffer is
        unchanged and this returns None instead.
        """
        newline_pos = self._buffer.find(b'\n')
        if newline_pos != -1:
            line = self._buffer[:newline_pos]
            self._buffer = self._buffer[newline_pos + 1:]
            return line
        else:
            return None

    def _state_expecting_format_line(self):
        line = self._consume_line()
        if line is not None:
            if line != FORMAT_ONE:
                raise UnknownContainerFormatError(line)
            self._state_handler = self._state_expecting_record_type

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

    def _state_expecting_length(self):
        line = self._consume_line()
        if line is not None:
            try:
                self._current_record_length = int(line)
            except ValueError:
                raise InvalidRecordError('{!r} is not a valid length.'.format(line))
            self._state_handler = self._state_expecting_name

    def _state_expecting_name(self):
        encoded_name_parts = self._consume_line()
        if encoded_name_parts == b'':
            self._state_handler = self._state_expecting_body
        elif encoded_name_parts:
            name_parts = tuple(encoded_name_parts.split(b'\x00'))
            for name_part in name_parts:
                _check_name(name_part)
            self._current_record_names.append(name_parts)

    def _state_expecting_body(self):
        if len(self._buffer) >= self._current_record_length:
            body_bytes = self._buffer[:self._current_record_length]
            self._buffer = self._buffer[self._current_record_length:]
            record = (self._current_record_names, body_bytes)
            self._parsed_records.append(record)
            self._reset_current_record()
            self._state_handler = self._state_expecting_record_type

    def _state_expecting_nothing(self):
        pass

    def read_size_hint(self):
        hint = 16384
        if self._state_handler == self._state_expecting_body:
            remaining = self._current_record_length - len(self._buffer)
            if remaining < 0:
                remaining = 0
            return max(hint, remaining)
        return hint