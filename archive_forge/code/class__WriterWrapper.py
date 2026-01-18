import csv
import datetime
import json
import logging
import apache_beam as beam
from six.moves import cStringIO
import yaml
from google.cloud.ml.util import _decoders
from google.cloud.ml.util import _file
class _WriterWrapper(object):
    """A wrapper for csv.writer / csv.DictWriter to make it picklable."""

    def __init__(self, column_names, delimiter, decode_to_dict):
        self._state = (column_names, delimiter, decode_to_dict)
        self._buffer = cStringIO()
        if decode_to_dict:
            self._writer = csv.DictWriter(self._buffer, column_names, lineterminator='', delimiter=delimiter)
        else:
            self._writer = csv.writer(self._buffer, lineterminator='', delimiter=delimiter)

    def encode_record(self, record):
        self._writer.writerow(record)
        value = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate(0)
        return value

    def __getstate__(self):
        return self._state

    def __setstate__(self, state):
        self.__init__(*state)