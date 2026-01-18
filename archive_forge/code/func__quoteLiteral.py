from __future__ import annotations
import codecs
import warnings
from typing import IO, TYPE_CHECKING, Optional, Tuple, Union
from rdflib.graph import Graph
from rdflib.serializer import Serializer
from rdflib.term import Literal
def _quoteLiteral(l_: Literal) -> str:
    """
    a simpler version of term.Literal.n3()
    """
    encoded = _quote_encode(l_)
    if l_.language:
        if l_.datatype:
            raise Exception('Literal has datatype AND language!')
        return '%s@%s' % (encoded, l_.language)
    elif l_.datatype:
        return '%s^^<%s>' % (encoded, l_.datatype)
    else:
        return '%s' % encoded