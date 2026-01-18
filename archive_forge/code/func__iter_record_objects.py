import re
from io import BytesIO
from .. import errors
def _iter_record_objects(self):
    while True:
        try:
            record_kind = self.reader_func(1)
        except StopIteration:
            return
        if record_kind == b'B':
            reader = BytesRecordReader(self._source)
            yield reader
        elif record_kind == b'E':
            return
        elif record_kind == b'':
            raise UnexpectedEndOfContainerError()
        else:
            raise UnknownRecordTypeError(record_kind)