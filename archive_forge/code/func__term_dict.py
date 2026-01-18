from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _term_dict(self, term: Term) -> Union[Dict[str, Any], str]:
    tdict: Dict[str, Any] = {}
    if term.type != UNDEF:
        tdict[TYPE] = self.shrink_iri(term.type)
    if term.container:
        tdict[CONTAINER] = list(term.container)
    if term.language != UNDEF:
        tdict[LANG] = term.language
    if term.reverse:
        tdict[REV] = term.id
    else:
        tdict[ID] = term.id
    if tdict.keys() == {ID}:
        return tdict[ID]
    return tdict