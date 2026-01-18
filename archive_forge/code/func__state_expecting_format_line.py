import re
from io import BytesIO
from .. import errors
def _state_expecting_format_line(self):
    line = self._consume_line()
    if line is not None:
        if line != FORMAT_ONE:
            raise UnknownContainerFormatError(line)
        self._state_handler = self._state_expecting_record_type