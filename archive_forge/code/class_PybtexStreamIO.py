from __future__ import absolute_import, unicode_literals
import pickle
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from io import BytesIO, TextIOWrapper
import six
import pytest
from pybtex.database import parse_bytes, parse_string, BibliographyData, Entry
from pybtex.plugin import find_plugin
from .data import reference_data
class PybtexStreamIO(PybtexDatabaseIO):

    def serialize(self, bib_data):
        stream = BytesIO()
        unicode_stream = TextIOWrapper(stream, 'UTF-8')
        self.writer.write_stream(bib_data, unicode_stream if self.writer.unicode_io else stream)
        unicode_stream.flush()
        stream.seek(0)
        return unicode_stream

    def deserialize(self, stream):
        parser_stream = stream if self.parser.unicode_io else stream.buffer
        return self.parser.parse_stream(parser_stream)