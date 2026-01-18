import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def _nodes_being_relocated(self, field):
    key, index, name_token = _unpack_key(field)
    nodes = self._kvpair_elements[key]
    nodes_being_relocated = []
    if name_token is not None or index is not None:
        single_node = self._resolve_to_single_node(nodes, key, index, name_token)
        assert single_node is not None
        nodes_being_relocated.append(single_node)
    else:
        nodes_being_relocated = nodes
    return (nodes, nodes_being_relocated)