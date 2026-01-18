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
def _create_input_source_from_location(file: Optional[Union[BinaryIO, TextIO]], format: Optional[str], input_source: Optional[InputSource], location: str) -> Tuple[URIRef, bool, Optional[Union[BinaryIO, TextIO]], Optional[InputSource]]:
    if os.path.exists(location):
        location = pathlib.Path(location).absolute().as_uri()
    base = pathlib.Path.cwd().as_uri()
    absolute_location = URIRef(rdflib.util._iri2uri(location), base=base)
    if absolute_location.startswith('file:///'):
        filename = url2pathname(absolute_location.replace('file:///', '/'))
        file = open(filename, 'rb')
    else:
        input_source = URLInputSource(absolute_location, format)
    auto_close = True
    return (absolute_location, auto_close, file, input_source)