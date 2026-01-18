import re
from io import BytesIO
from .. import errors
def _state_expecting_length(self):
    line = self._consume_line()
    if line is not None:
        try:
            self._current_record_length = int(line)
        except ValueError:
            raise InvalidRecordError('{!r} is not a valid length.'.format(line))
        self._state_handler = self._state_expecting_name