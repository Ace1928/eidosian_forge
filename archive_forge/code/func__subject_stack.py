from contextlib import contextmanager
from rdflib.graph import Graph
from rdflib.namespace import RDF
from rdflib.term import BNode, Identifier, Literal, URIRef
@contextmanager
def _subject_stack(self, subject):
    self._subjects.append(subject)
    yield None
    self._subjects.pop()