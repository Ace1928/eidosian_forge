from __future__ import annotations
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from operator import methodcaller
from typing import TYPE_CHECKING
from urllib.parse import unquote, urldefrag, urljoin, urlsplit
from urllib.request import urlopen
from warnings import warn
import contextlib
import json
import reprlib
import warnings
from attrs import define, field, fields
from jsonschema_specifications import REGISTRY as SPECIFICATIONS
from rpds import HashTrieMap
import referencing.exceptions
import referencing.jsonschema
from jsonschema import (
def _validate_reference(self, ref, instance):
    if self._ref_resolver is None:
        try:
            resolved = self._resolver.lookup(ref)
        except referencing.exceptions.Unresolvable as err:
            raise exceptions._WrappedReferencingError(err) from err
        return self.descend(instance, resolved.contents, resolver=resolved.resolver)
    else:
        resolve = getattr(self._ref_resolver, 'resolve', None)
        if resolve is None:
            with self._ref_resolver.resolving(ref) as resolved:
                return self.descend(instance, resolved)
        else:
            scope, resolved = resolve(ref)
            self._ref_resolver.push_scope(scope)
            try:
                return list(self.descend(instance, resolved))
            finally:
                self._ref_resolver.pop_scope()