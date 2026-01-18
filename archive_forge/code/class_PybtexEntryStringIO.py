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
class PybtexEntryStringIO(PybtexDatabaseIO):

    def __init__(self, bib_format):
        super(PybtexEntryStringIO, self).__init__(bib_format)
        self.key = list(reference_data.entries.keys())[0]
        self.reference_data = reference_data.entries[self.key]
        assert reference_data.entries
        assert reference_data.preamble

    def serialize(self, bib_data):
        result = bib_data.to_string(self.bib_format)
        assert isinstance(result, six.text_type)
        return result

    def deserialize(self, string):
        return Entry.from_string(string, self.bib_format)