import re
from io import BytesIO
from .. import errors
def _state_expecting_name(self):
    encoded_name_parts = self._consume_line()
    if encoded_name_parts == b'':
        self._state_handler = self._state_expecting_body
    elif encoded_name_parts:
        name_parts = tuple(encoded_name_parts.split(b'\x00'))
        for name_part in name_parts:
            _check_name(name_part)
        self._current_record_names.append(name_parts)