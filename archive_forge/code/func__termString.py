from typing import IO, List, Optional, Union
from rdflib.namespace import NamespaceManager
from rdflib.query import ResultSerializer
from rdflib.term import BNode, Literal, URIRef, Variable
def _termString(t: Optional[Union[URIRef, Literal, BNode]], namespace_manager: Optional[NamespaceManager]) -> str:
    if t is None:
        return '-'
    if namespace_manager:
        if isinstance(t, URIRef):
            return namespace_manager.normalizeUri(t)
        elif isinstance(t, BNode):
            return t.n3()
        elif isinstance(t, Literal):
            return t._literal_n3(qname_callback=namespace_manager.normalizeUri)
    else:
        return t.n3()