import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def CastClass(c, graph=None):
    graph = graph is None and c.factoryGraph or graph
    for kind in graph.objects(subject=classOrIdentifier(c), predicate=RDF.type):
        if kind == OWL.Restriction:
            kwargs = {'identifier': classOrIdentifier(c), 'graph': graph}
            for _s, p, o in graph.triples((classOrIdentifier(c), None, None)):
                if p != RDF.type:
                    if p == OWL.onProperty:
                        kwargs['onProperty'] = o
                    else:
                        if p not in Restriction.restrictionKinds:
                            continue
                        kwargs[str(p.split(str(OWL))[-1])] = o
            if not set([str(i.split(str(OWL))[-1]) for i in Restriction.restrictionKinds]).intersection(kwargs):
                raise MalformedClassError('Malformed owl:Restriction')
            return Restriction(**kwargs)
        else:
            for _s, p, _o in graph.triples_choices((classOrIdentifier(c), [OWL.intersectionOf, OWL.unionOf, OWL.oneOf], None)):
                if p == OWL.oneOf:
                    return EnumeratedClass(classOrIdentifier(c), graph=graph)
                else:
                    return BooleanClass(classOrIdentifier(c), operator=p, graph=graph)
            return Class(classOrIdentifier(c), graph=graph, skipOWLClassMembership=True)