import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
class Individual:
    """
    A typed individual, the base class of the InfixOWL classes.

    """
    factoryGraph = Graph()

    def serialize(self, graph):
        for fact in self.factoryGraph.triples((self.identifier, None, None)):
            graph.add(fact)

    def __init__(self, identifier=None, graph=None):
        self.__identifier = identifier is not None and identifier or BNode()
        if graph is None:
            self.graph = self.factoryGraph
        else:
            self.graph = graph
        self.qname = None
        if not isinstance(self.identifier, BNode):
            try:
                prefix, uri, localname = self.graph.compute_qname(self.identifier)
                self.qname = ':'.join([prefix, localname])
            except Exception:
                pass

    def clearInDegree(self):
        """
        Remove references to this individual as an object in the
        backing store.
        """
        self.graph.remove((None, None, self.identifier))

    def clearOutDegree(self):
        """
        Remove all statements to this individual as a subject in the
        backing store. Note that this only removes the statements
        themselves, not the blank node closure so there is a chance
        that this will cause orphaned blank nodes to remain in the
        graph.
        """
        self.graph.remove((self.identifier, None, None))

    def delete(self):
        """
        Delete the individual from the graph, clearing the in and
        out degrees.
        """
        self.clearInDegree()
        self.clearOutDegree()

    def replace(self, other):
        """
        Replace the individual in the graph with the given other,
        causing all triples that refer to it to be changed and then
        delete the individual.

        >>> g = Graph()
        >>> b = Individual(OWL.Restriction, g)
        >>> b.type = RDFS.Resource
        >>> len(list(b.type))
        1
        >>> del b.type
        >>> len(list(b.type))
        0
        """
        for s, p, _o in self.graph.triples((None, None, self.identifier)):
            self.graph.add((s, p, classOrIdentifier(other)))
        self.delete()

    def _get_type(self):
        for _t in self.graph.objects(subject=self.identifier, predicate=RDF.type):
            yield _t

    def _set_type(self, kind):
        if not kind:
            return
        if isinstance(kind, (Individual, Identifier)):
            self.graph.add((self.identifier, RDF.type, classOrIdentifier(kind)))
        else:
            for c in kind:
                assert isinstance(c, (Individual, Identifier))
                self.graph.add((self.identifier, RDF.type, classOrIdentifier(c)))

    @TermDeletionHelper(RDF.type)
    def _delete_type(self):
        """
        >>> g = Graph()
        >>> b = Individual(OWL.Restriction, g)
        >>> b.type = RDFS.Resource
        >>> len(list(b.type))
        1
        >>> del b.type
        >>> len(list(b.type))
        0
        """
        pass
    type = property(_get_type, _set_type, _delete_type)

    def _get_identifier(self):
        return self.__identifier

    def _set_identifier(self, i):
        assert i
        if i != self.__identifier:
            oldstatements_out = [(p, o) for s, p, o in self.graph.triples((self.__identifier, None, None))]
            oldstatements_in = [(s, p) for s, p, o in self.graph.triples((None, None, self.__identifier))]
            for p1, o1 in oldstatements_out:
                self.graph.remove((self.__identifier, p1, o1))
            for s1, p1 in oldstatements_in:
                self.graph.remove((s1, p1, self.__identifier))
            self.__identifier = i
            self.graph.addN([(i, p1, o1, self.graph) for p1, o1 in oldstatements_out])
            self.graph.addN([(s1, p1, i, self.graph) for s1, p1 in oldstatements_in])
        if not isinstance(i, BNode):
            try:
                prefix, uri, localname = self.graph.compute_qname(i)
                self.qname = ':'.join([prefix, localname])
            except Exception:
                pass
    identifier = property(_get_identifier, _set_identifier)

    def _get_sameAs(self):
        for _t in self.graph.objects(subject=self.identifier, predicate=OWL.sameAs):
            yield _t

    def _set_sameAs(self, term):
        if isinstance(term, (Individual, Identifier)):
            self.graph.add((self.identifier, OWL.sameAs, classOrIdentifier(term)))
        else:
            for c in term:
                assert isinstance(c, (Individual, Identifier))
                self.graph.add((self.identifier, OWL.sameAs, classOrIdentifier(c)))

    @TermDeletionHelper(OWL.sameAs)
    def _delete_sameAs(self):
        pass
    sameAs = property(_get_sameAs, _set_sameAs, _delete_sameAs)