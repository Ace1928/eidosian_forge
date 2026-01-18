import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
class Ontology(AnnotatableTerms):
    """The owl ontology metadata"""

    def __init__(self, identifier=None, imports=None, comment=None, graph=None):
        super(Ontology, self).__init__(identifier, graph)
        self.imports = [] if imports is None else imports
        self.comment = [] if comment is None else comment
        if (self.identifier, RDF.type, OWL.Ontology) not in self.graph:
            self.graph.add((self.identifier, RDF.type, OWL.Ontology))

    def setVersion(self, version):
        self.graph.set((self.identifier, OWL.versionInfo, version))

    def _get_imports(self):
        for owl in self.graph.objects(subject=self.identifier, predicate=OWL['imports']):
            yield owl

    def _set_imports(self, other):
        if not other:
            return
        for o in other:
            self.graph.add((self.identifier, OWL['imports'], o))

    @TermDeletionHelper(OWL['imports'])
    def _del_imports(self):
        pass
    imports = property(_get_imports, _set_imports, _del_imports)