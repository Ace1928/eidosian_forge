from __future__ import absolute_import, print_function, division
import io
import json
import inspect
from json.encoder import JSONEncoder
from os import unlink
from tempfile import NamedTemporaryFile
from petl.compat import PY2
from petl.compat import pickle
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.util.base import data, Table, dicts as _dicts, iterpeek
class JsonView(Table):

    def __init__(self, source, *args, **kwargs):
        self.source = source
        self.missing = kwargs.pop('missing', None)
        self.header = kwargs.pop('header', None)
        self.sample = kwargs.pop('sample', 1000)
        self.lines = kwargs.pop('lines', False)
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        with self.source.open('rb') as f:
            if not PY2:
                f = io.TextIOWrapper(f, encoding='utf-8', newline='', write_through=True)
            try:
                if self.lines:
                    for row in iterjlines(f, self.header, self.missing):
                        yield row
                else:
                    dicts = json.load(f, *self.args, **self.kwargs)
                    for row in iterdicts(dicts, self.header, self.sample, self.missing):
                        yield row
            finally:
                if not PY2:
                    f.detach()