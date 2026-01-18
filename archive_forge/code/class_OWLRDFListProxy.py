import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
class OWLRDFListProxy:

    def __init__(self, rdf_list, members=None, graph=None):
        if graph:
            self.graph = graph
        members = [] if members is None else members
        if rdf_list:
            self._rdfList = Collection(self.graph, rdf_list[0])
            for member in members:
                if member not in self._rdfList:
                    self._rdfList.append(classOrIdentifier(member))
        else:
            self._rdfList = Collection(self.graph, BNode(), [classOrIdentifier(m) for m in members])
            self.graph.add((self.identifier, self._operator, self._rdfList.uri))

    def __eq__(self, other):
        """
        Equivalence of boolean class constructors is determined by
        equivalence of its members
        """
        assert isinstance(other, Class), repr(other) + repr(type(other))
        if isinstance(other, BooleanClass):
            length = len(self)
            if length != len(other):
                return False
            else:
                for idx in range(length):
                    if self[idx] != other[idx]:
                        return False
                    return True
        else:
            return self.identifier == other.identifier

    def __len__(self):
        return len(self._rdfList)

    def index(self, item):
        return self._rdfList.index(classOrIdentifier(item))

    def __getitem__(self, key):
        return self._rdfList[key]

    def __setitem__(self, key, value):
        self._rdfList[key] = classOrIdentifier(value)

    def __delitem__(self, key):
        del self._rdfList[key]

    def clear(self):
        self._rdfList.clear()

    def __iter__(self):
        for item in self._rdfList:
            yield item

    def __contains__(self, item):
        for i in self._rdfList:
            if i == classOrIdentifier(item):
                return 1
        return 0

    def append(self, item):
        self._rdfList.append(item)

    def __iadd__(self, other):
        self._rdfList.append(classOrIdentifier(other))
        return self