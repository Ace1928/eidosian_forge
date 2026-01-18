import re
from fractions import Fraction
import logging
import math
import warnings
import xml.dom.minidom
from base64 import b64decode, b64encode
from binascii import hexlify, unhexlify
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from re import compile, sub
from typing import (
from urllib.parse import urldefrag, urljoin, urlparse
from isodate import (
import rdflib
import rdflib.util
from rdflib.compat import long_type
class URIRef(IdentifiedNode):
    """
    RDF 1.1's IRI Section https://www.w3.org/TR/rdf11-concepts/#section-IRIs

    .. note:: Documentation on RDF outside of RDFLib uses the term IRI or URI whereas this class is called URIRef. This is because it was made when the first version of the RDF specification was current, and it used the term *URIRef*, see `RDF 1.0 URIRef <http://www.w3.org/TR/rdf-concepts/#section-Graph-URIref>`_

    An IRI (Internationalized Resource Identifier) within an RDF graph is a Unicode string that conforms to the syntax defined in RFC 3987.

    IRIs in the RDF abstract syntax MUST be absolute, and MAY contain a fragment identifier.

    IRIs are a generalization of URIs [RFC3986] that permits a wider range of Unicode characters.
    """
    __slots__ = ()
    __or__: Callable[['URIRef', Union['URIRef', 'Path']], 'AlternativePath']
    __invert__: Callable[['URIRef'], 'InvPath']
    __neg__: Callable[['URIRef'], 'NegatedPath']
    __truediv__: Callable[['URIRef', Union['URIRef', 'Path']], 'SequencePath']

    def __new__(cls, value: str, base: Optional[str]=None) -> 'URIRef':
        if base is not None:
            ends_in_hash = value.endswith('#')
            value = urljoin(base, value, allow_fragments=1)
            if ends_in_hash:
                if not value.endswith('#'):
                    value += '#'
        if not _is_valid_uri(value):
            logger.warning('%s does not look like a valid URI, trying to serialize this will break.' % value)
        try:
            rt = str.__new__(cls, value)
        except UnicodeDecodeError:
            rt = str.__new__(cls, value, 'utf-8')
        return rt

    def n3(self, namespace_manager: Optional['NamespaceManager']=None) -> str:
        """
        This will do a limited check for valid URIs,
        essentially just making sure that the string includes no illegal
        characters (``<, >, ", {, }, |, \\, `, ^``)

        :param namespace_manager: if not None, will be used to make up
             a prefixed name
        """
        if not _is_valid_uri(self):
            raise Exception('"%s" does not look like a valid URI, I cannot serialize this as N3/Turtle. Perhaps you wanted to urlencode it?' % self)
        if namespace_manager:
            return namespace_manager.normalizeUri(self)
        else:
            return '<%s>' % self

    def defrag(self) -> 'URIRef':
        if '#' in self:
            url, frag = urldefrag(self)
            return URIRef(url)
        else:
            return self

    @property
    def fragment(self) -> str:
        """
        Return the URL Fragment

        >>> URIRef("http://example.com/some/path/#some-fragment").fragment
        'some-fragment'
        >>> URIRef("http://example.com/some/path/").fragment
        ''
        """
        return urlparse(self).fragment

    def __reduce__(self) -> Tuple[Type['URIRef'], Tuple[str]]:
        return (URIRef, (str(self),))

    def __repr__(self) -> str:
        if self.__class__ is URIRef:
            clsName = 'rdflib.term.URIRef'
        else:
            clsName = self.__class__.__name__
        return '%s(%s)' % (clsName, super(URIRef, self).__repr__())

    def __add__(self, other) -> 'URIRef':
        return self.__class__(str(self) + other)

    def __radd__(self, other) -> 'URIRef':
        return self.__class__(other + str(self))

    def __mod__(self, other) -> 'URIRef':
        return self.__class__(str(self) % other)

    def de_skolemize(self) -> 'BNode':
        """Create a Blank Node from a skolem URI, in accordance
        with http://www.w3.org/TR/rdf11-concepts/#section-skolemization.
        This function accepts only rdflib type skolemization, to provide
        a round-tripping within the system.

        .. versionadded:: 4.0
        """
        if isinstance(self, RDFLibGenid):
            parsed_uri = urlparse('%s' % self)
            return BNode(value=parsed_uri.path[len(rdflib_skolem_genid):])
        elif isinstance(self, Genid):
            bnode_id = '%s' % self
            if bnode_id in skolems:
                return skolems[bnode_id]
            else:
                retval = BNode()
                skolems[bnode_id] = retval
                return retval
        else:
            raise Exception('<%s> is not a skolem URI' % self)