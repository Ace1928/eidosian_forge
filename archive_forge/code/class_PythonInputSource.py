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
class PythonInputSource(InputSource):
    '''
    Constructs an RDFLib Parser InputSource from a Python data structure,
    for example, loaded from JSON with json.load or json.loads:

    >>> import json
    >>> as_string = """{
    ...   "@context" : {"ex" : "http://example.com/ns#"},
    ...   "@graph": [{"@type": "ex:item", "@id": "#example"}]
    ... }"""
    >>> as_python = json.loads(as_string)
    >>> source = create_input_source(data=as_python)
    >>> isinstance(source, PythonInputSource)
    True
    '''

    def __init__(self, data: Any, system_id: Optional[str]=None):
        self.content_type = None
        self.auto_close = False
        self.public_id: Optional[str] = None
        self.system_id: Optional[str] = system_id
        self.data = data

    def getPublicId(self) -> Optional[str]:
        return self.public_id

    def setPublicId(self, public_id: Optional[str]) -> None:
        self.public_id = public_id

    def getSystemId(self) -> Optional[str]:
        return self.system_id

    def setSystemId(self, system_id: Optional[str]) -> None:
        self.system_id = system_id

    def close(self) -> None:
        self.data = None