import xml.dom.minidom
from typing import IO, Dict, Optional, Set
from xml.sax.saxutils import escape, quoteattr
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import RDF, RDFS, Namespace  # , split_uri
from rdflib.plugins.parsers.RDFVOC import RDFVOC
from rdflib.plugins.serializers.xmlwriter import XMLWriter
from rdflib.serializer import Serializer
from rdflib.term import BNode, IdentifiedNode, Identifier, Literal, Node, URIRef
from rdflib.util import first, more_than
from .xmlwriter import ESCAPE_ENTITIES
class PrettyXMLSerializer(Serializer):

    def __init__(self, store: Graph, max_depth=3):
        super(PrettyXMLSerializer, self).__init__(store)
        self.forceRDFAbout: Set[URIRef] = set()

    def serialize(self, stream: IO[bytes], base: Optional[str]=None, encoding: Optional[str]=None, **args):
        self.__serialized: Dict[Identifier, int] = {}
        store = self.store
        if base is not None:
            self.base = base
        elif store.base is not None:
            self.base = store.base
        self.max_depth = args.get('max_depth', 3)
        assert self.max_depth > 0, 'max_depth must be greater than 0'
        self.nm = nm = store.namespace_manager
        self.writer = writer = XMLWriter(stream, nm, encoding)
        namespaces = {}
        possible: Set[Node] = set(store.predicates()).union(store.objects(None, RDF.type))
        for predicate in possible:
            prefix, namespace, local = nm.compute_qname_strict(predicate)
            namespaces[prefix] = namespace
        namespaces['rdf'] = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        writer.push(RDFVOC.RDF)
        if 'xml_base' in args:
            writer.attribute(XMLBASE, args['xml_base'])
        elif self.base:
            writer.attribute(XMLBASE, self.base)
        writer.namespaces(namespaces.items())
        subject: IdentifiedNode
        for subject in store.subjects():
            if (None, None, subject) in store:
                if (subject, None, subject) in store:
                    self.subject(subject, 1)
            else:
                self.subject(subject, 1)
        bnodes = set()
        for subject in store.subjects():
            if isinstance(subject, BNode):
                bnodes.add(subject)
                continue
            self.subject(subject, 1)
        for bnode in bnodes:
            if bnode not in self.__serialized:
                self.subject(subject, 1)
        writer.pop(RDFVOC.RDF)
        stream.write('\n'.encode('latin-1'))
        self.__serialized = None

    def subject(self, subject: IdentifiedNode, depth: int=1):
        store = self.store
        writer = self.writer
        if subject in self.forceRDFAbout:
            writer.push(RDFVOC.Description)
            writer.attribute(RDFVOC.about, self.relativize(subject))
            writer.pop(RDFVOC.Description)
            self.forceRDFAbout.remove(subject)
        elif subject not in self.__serialized:
            self.__serialized[subject] = 1
            type = first(store.objects(subject, RDF.type))
            try:
                self.nm.qname(type)
            except Exception:
                type = None
            element = type or RDFVOC.Description
            writer.push(element)
            if isinstance(subject, BNode):

                def subj_as_obj_more_than(ceil):
                    return True
                if subj_as_obj_more_than(1):
                    writer.attribute(RDFVOC.nodeID, fix(subject))
            else:
                writer.attribute(RDFVOC.about, self.relativize(subject))
            if (subject, None, None) in store:
                for predicate, object in store.predicate_objects(subject):
                    if not (predicate == RDF.type and object == type):
                        self.predicate(predicate, object, depth + 1)
            writer.pop(element)
        elif subject in self.forceRDFAbout:
            writer.push(RDFVOC.Description)
            writer.attribute(RDFVOC.about, self.relativize(subject))
            writer.pop(RDFVOC.Description)
            self.forceRDFAbout.remove(subject)

    def predicate(self, predicate, object, depth=1):
        writer = self.writer
        store = self.store
        writer.push(predicate)
        if isinstance(object, Literal):
            if object.language:
                writer.attribute(XMLLANG, object.language)
            if object.datatype == RDF.XMLLiteral and isinstance(object.value, xml.dom.minidom.Document):
                writer.attribute(RDFVOC.parseType, 'Literal')
                writer.text('')
                writer.stream.write(object)
            else:
                if object.datatype:
                    writer.attribute(RDFVOC.datatype, object.datatype)
                writer.text(object)
        elif object in self.__serialized or not (object, None, None) in store:
            if isinstance(object, BNode):
                if more_than(store.triples((None, None, object)), 0):
                    writer.attribute(RDFVOC.nodeID, fix(object))
            else:
                writer.attribute(RDFVOC.resource, self.relativize(object))
        elif first(store.objects(object, RDF.first)):
            self.__serialized[object] = 1
            import warnings
            warnings.warn('Assertions on %s other than RDF.first ' % repr(object) + 'and RDF.rest are ignored ... including RDF.List', UserWarning, stacklevel=2)
            writer.attribute(RDFVOC.parseType, 'Collection')
            col = Collection(store, object)
            for item in col:
                if isinstance(item, URIRef):
                    self.forceRDFAbout.add(item)
                self.subject(item)
                if not isinstance(item, URIRef):
                    self.__serialized[item] = 1
        elif first(store.triples_choices((object, RDF.type, [OWL_NS.Class, RDFS.Class]))) and isinstance(object, URIRef):
            writer.attribute(RDFVOC.resource, self.relativize(object))
        elif depth <= self.max_depth:
            self.subject(object, depth + 1)
        elif isinstance(object, BNode):
            if object not in self.__serialized and (object, None, None) in store and (len(list(store.subjects(object=object))) == 1):
                self.subject(object, depth + 1)
            else:
                writer.attribute(RDFVOC.nodeID, fix(object))
        else:
            writer.attribute(RDFVOC.resource, self.relativize(object))
        writer.pop(predicate)