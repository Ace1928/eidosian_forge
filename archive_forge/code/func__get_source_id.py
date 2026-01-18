from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _get_source_id(self, source: Dict[str, Any], key: str) -> Optional[str]:
    term = source.get(key)
    if term is None:
        dfn = self.terms.get(key)
        if dfn:
            term = dfn.id
    elif isinstance(term, dict):
        term = term.get(ID)
    return term