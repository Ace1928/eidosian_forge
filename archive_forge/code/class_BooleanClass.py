import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
class BooleanClass(OWLRDFListProxy, Class):
    """
    See: http://www.w3.org/TR/owl-ref/#Boolean

    owl:complementOf is an attribute of Class, however

    """

    @BooleanClassExtentHelper(OWL.intersectionOf)
    @Callable
    def getIntersections():
        pass
    getIntersections = Callable(getIntersections)

    @BooleanClassExtentHelper(OWL.unionOf)
    @Callable
    def getUnions():
        pass
    getUnions = Callable(getUnions)

    def __init__(self, identifier=None, operator=OWL.intersectionOf, members=None, graph=None):
        if operator is None:
            props = []
            for _s, p, _o in graph.triples_choices((identifier, [OWL.intersectionOf, OWL.unionOf], None)):
                props.append(p)
                operator = p
            assert len(props) == 1, repr(props)
        Class.__init__(self, identifier, graph=graph)
        assert operator in [OWL.intersectionOf, OWL.unionOf], str(operator)
        self._operator = operator
        rdf_list = list(self.graph.objects(predicate=operator, subject=self.identifier))
        assert not members or not rdf_list, 'This is a previous boolean class description.'
        OWLRDFListProxy.__init__(self, rdf_list, members)

    def copy(self):
        """
        Create a copy of this class
        """
        copy_of_class = BooleanClass(operator=self._operator, members=list(self), graph=self.graph)
        return copy_of_class

    def serialize(self, graph):
        clonedlist = Collection(graph, BNode())
        for cl in self._rdfList:
            clonedlist.append(cl)
            CastClass(cl, self.graph).serialize(graph)
        graph.add((self.identifier, self._operator, clonedlist.uri))
        for s, p, o in self.graph.triples((self.identifier, None, None)):
            if p != self._operator:
                graph.add((s, p, o))
        self._serialize(graph)

    def isPrimitive(self):
        return False

    def changeOperator(self, newOperator):
        """
        Converts a unionOf / intersectionOf class expression into one
        that instead uses the given operator

        >>> testGraph = Graph()
        >>> Individual.factoryGraph = testGraph
        >>> EX = Namespace("http://example.com/")
        >>> testGraph.bind("ex", EX, override=False)
        >>> fire = Class(EX.Fire)
        >>> water = Class(EX.Water)
        >>> testClass = BooleanClass(members=[fire,water])
        >>> testClass
        ( ex:Fire AND ex:Water )
        >>> testClass.changeOperator(OWL.unionOf)
        >>> testClass
        ( ex:Fire OR ex:Water )
        >>> try:
        ...     testClass.changeOperator(OWL.unionOf)
        ... except Exception as e:
        ...     print(e)  # doctest: +SKIP
        The new operator is already being used!

        """
        assert newOperator != self._operator, 'The new operator is already being used!'
        self.graph.remove((self.identifier, self._operator, self._rdfList.uri))
        self.graph.add((self.identifier, newOperator, self._rdfList.uri))
        self._operator = newOperator

    def __repr__(self):
        """
        Returns the Manchester Syntax equivalent for this class
        """
        return manchesterSyntax(self._rdfList.uri if isinstance(self._rdfList, Collection) else BNode(), self.graph, boolean=self._operator)

    def __or__(self, other):
        """
        Adds other to the list and returns self
        """
        assert self._operator == OWL.unionOf
        self._rdfList.append(classOrIdentifier(other))
        return self