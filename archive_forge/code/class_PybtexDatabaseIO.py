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
class PybtexDatabaseIO(DatabaseIO):

    def __init__(self, bib_format):
        super(PybtexDatabaseIO, self).__init__()
        self.bib_format = bib_format
        self.writer = find_plugin('pybtex.database.output', bib_format)(encoding='UTF-8')
        self.parser = find_plugin('pybtex.database.input', bib_format)(encoding='UTF-8')
        if bib_format == 'bibtexml':
            self.reference_data._preamble = []

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.bib_format)