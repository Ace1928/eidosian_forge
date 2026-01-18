import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def ComponentTerms(cls):
    """
    Takes a Class instance and returns a generator over the classes that
    are involved in its definition, ignoring unnamed classes
    """
    if OWL.Restriction in cls.type:
        try:
            cls = CastClass(cls, Individual.factoryGraph)
            for _s, _p, inner_class_id in cls.factoryGraph.triples_choices((cls.identifier, [OWL.allValuesFrom, OWL.someValuesFrom], None)):
                inner_class = Class(inner_class_id, skipOWLClassMembership=True)
                if isinstance(inner_class_id, BNode):
                    for _c in ComponentTerms(inner_class):
                        yield _c
                else:
                    yield inner_class
        except Exception:
            pass
    else:
        cls = CastClass(cls, Individual.factoryGraph)
        if isinstance(cls, BooleanClass):
            for _cls in cls:
                _cls = Class(_cls, skipOWLClassMembership=True)
                if isinstance(_cls.identifier, BNode):
                    for _c in ComponentTerms(_cls):
                        yield _c
                else:
                    yield _cls
        else:
            for inner_class in cls.subClassOf:
                if isinstance(inner_class.identifier, BNode):
                    for _c in ComponentTerms(inner_class):
                        yield _c
                else:
                    yield inner_class
            for _s, _p, o in cls.factoryGraph.triples_choices((classOrIdentifier(cls), CLASS_RELATIONS, None)):
                if isinstance(o, BNode):
                    for _c in ComponentTerms(CastClass(o, Individual.factoryGraph)):
                        yield _c
                else:
                    yield inner_class