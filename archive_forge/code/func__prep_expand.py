from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def _prep_expand(self, expr: str) -> Tuple[bool, Optional[str], str]:
    if ':' not in expr:
        return (True, None, expr)
    pfx, local = expr.split(':', 1)
    if not local.startswith('//'):
        return (False, pfx, local)
    else:
        return (False, None, expr)