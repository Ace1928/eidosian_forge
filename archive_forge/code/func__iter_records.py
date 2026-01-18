import re
from io import BytesIO
from .. import errors
def _iter_records(self):
    for record in self._iter_record_objects():
        yield record.read()