from __future__ import annotations
import codecs
import warnings
from typing import IO, TYPE_CHECKING, Optional, Tuple, Union
from rdflib.graph import Graph
from rdflib.serializer import Serializer
from rdflib.term import Literal
def _nt_unicode_error_resolver(err: UnicodeError) -> Tuple[Union[str, bytes], int]:
    """
    Do unicode char replaces as defined in https://www.w3.org/TR/2004/REC-rdf-testcases-20040210/#ntrip_strings
    """

    def _replace_single(c):
        c = ord(c)
        fmt = '\\u%04X' if c <= 65535 else '\\U%08X'
        return fmt % c
    string = err.object[err.start:err.end]
    return (''.join((_replace_single(c) for c in string)), err.end)