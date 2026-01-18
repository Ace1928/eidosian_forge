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
class PybtexBytesIO(PybtexDatabaseIO):

    def serialize(self, bib_data):
        result = bib_data.to_bytes(self.bib_format)
        assert isinstance(result, bytes)
        return result

    def deserialize(self, string):
        return parse_bytes(string, self.bib_format)