from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _read_source(self, source: Dict[str, Any], source_url: Optional[str]=None, referenced_contexts: Optional[Set[str]]=None):
    imports = source.get(IMPORT)
    if imports:
        if not isinstance(imports, str):
            raise INVALID_CONTEXT_ENTRY
        imported = self._fetch_context(imports, self.base, referenced_contexts or set())
        if not isinstance(imported, dict):
            raise INVALID_CONTEXT_ENTRY
        imported = imported[CONTEXT]
        imported.update(source)
        source = imported
    self.vocab = source.get(VOCAB, self.vocab)
    self.version = source.get(VERSION, self.version)
    protected = source.get(PROTECTED, False)
    for key, value in source.items():
        if key in {VOCAB, VERSION, IMPORT, PROTECTED}:
            continue
        elif key == PROPAGATE and isinstance(value, bool):
            self.propagate = value
        elif key == LANG:
            self.language = value
        elif key == BASE:
            if not source_url and (not imports):
                self.base = value
        else:
            self._read_term(source, key, value, protected)