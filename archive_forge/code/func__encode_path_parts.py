import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _encode_path_parts(text_parts, rooted=False, has_scheme=True, has_authority=True, maximal=True):
    """
    Percent-encode a tuple of path parts into a complete path.

    Setting *maximal* to False percent-encodes only the reserved
    characters that are syntactically necessary for serialization,
    preserving any IRI-style textual data.

    Leaving *maximal* set to its default True percent-encodes
    everything required to convert a portion of an IRI to a portion of
    a URI.

    RFC 3986 3.3:

       If a URI contains an authority component, then the path component
       must either be empty or begin with a slash ("/") character.  If a URI
       does not contain an authority component, then the path cannot begin
       with two slash characters ("//").  In addition, a URI reference
       (Section 4.1) may be a relative-path reference, in which case the
       first path segment cannot contain a colon (":") character.
    """
    if not text_parts:
        return ()
    if rooted:
        text_parts = (u'',) + tuple(text_parts)
    encoded_parts = []
    if has_scheme:
        encoded_parts = [_encode_path_part(part, maximal=maximal) if part else part for part in text_parts]
    else:
        encoded_parts = [_encode_schemeless_path_part(text_parts[0])]
        encoded_parts.extend([_encode_path_part(part, maximal=maximal) if part else part for part in text_parts[1:]])
    return tuple(encoded_parts)