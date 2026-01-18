from __future__ import annotations
import codecs
import os
import pathlib
import sys
from io import BufferedIOBase, BytesIO, RawIOBase, StringIO, TextIOBase, TextIOWrapper
from typing import (
from urllib.parse import urljoin
from urllib.request import Request, url2pathname
from xml.sax import xmlreader
import rdflib.util
from rdflib import __version__
from rdflib._networking import _urlopen
from rdflib.namespace import Namespace
from rdflib.term import URIRef
class BytesIOWrapper(BufferedIOBase):
    __slots__ = ('wrapped', 'encoded', 'encoding')

    def __init__(self, wrapped: str, encoding='utf-8'):
        super(BytesIOWrapper, self).__init__()
        self.wrapped = wrapped
        self.encoding = encoding
        self.encoded = None

    def read(self, *args, **kwargs):
        if self.encoded is None:
            b, blen = codecs.getencoder(self.encoding)(self.wrapped)
            self.encoded = BytesIO(b)
        return self.encoded.read(*args, **kwargs)

    def read1(self, *args, **kwargs):
        if self.encoded is None:
            b = codecs.getencoder(self.encoding)(self.wrapped)
            self.encoded = BytesIO(b)
        return self.encoded.read1(*args, **kwargs)

    def readinto(self, *args, **kwargs):
        raise NotImplementedError()

    def readinto1(self, *args, **kwargs):
        raise NotImplementedError()

    def write(self, *args, **kwargs):
        raise NotImplementedError()