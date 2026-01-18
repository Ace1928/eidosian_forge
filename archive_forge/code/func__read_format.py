import re
from io import BytesIO
from .. import errors
def _read_format(self):
    format = self._read_line()
    if format != FORMAT_ONE:
        raise UnknownContainerFormatError(format)