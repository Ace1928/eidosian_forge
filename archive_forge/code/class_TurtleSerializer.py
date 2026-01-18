from collections import defaultdict
from functools import cmp_to_key
from rdflib.exceptions import Error
from rdflib.namespace import RDF, RDFS
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
class TurtleSerializer(RecursiveSerializer):
    short_name = 'turtle'
    indentString = '    '

    def __init__(self, store):
        self._ns_rewrite = {}
        super(TurtleSerializer, self).__init__(store)
        self.keywords = {RDF.type: 'a'}
        self.reset()
        self.stream = None
        self._spacious = _SPACIOUS_OUTPUT

    def addNamespace(self, prefix, namespace):
        if prefix > '' and prefix[0] == '_' or self.namespaces.get(prefix, namespace) != namespace:
            if prefix not in self._ns_rewrite:
                p = 'p' + prefix
                while p in self.namespaces:
                    p = 'p' + p
                self._ns_rewrite[prefix] = p
            prefix = self._ns_rewrite.get(prefix, prefix)
        super(TurtleSerializer, self).addNamespace(prefix, namespace)
        return prefix

    def reset(self):
        super(TurtleSerializer, self).reset()
        self._shortNames = {}
        self._started = False
        self._ns_rewrite = {}

    def serialize(self, stream, base=None, encoding=None, spacious=None, **args):
        self.reset()
        self.stream = stream
        if base is not None:
            self.base = base
        elif self.store.base is not None:
            self.base = self.store.base
        if spacious is not None:
            self._spacious = spacious
        self.preprocess()
        subjects_list = self.orderSubjects()
        self.startDocument()
        firstTime = True
        for subject in subjects_list:
            if self.isDone(subject):
                continue
            if firstTime:
                firstTime = False
            if self.statement(subject) and (not firstTime):
                self.write('\n')
        self.endDocument()
        stream.write('\n'.encode('latin-1'))
        self.base = None

    def preprocessTriple(self, triple):
        super(TurtleSerializer, self).preprocessTriple(triple)
        for i, node in enumerate(triple):
            if i == VERB and node in self.keywords:
                continue
            self.getQName(node, gen_prefix=i == VERB)
            if isinstance(node, Literal) and node.datatype:
                self.getQName(node.datatype, gen_prefix=_GEN_QNAME_FOR_DT)
        p = triple[1]
        if isinstance(p, BNode):
            self._references[p] += 1

    def getQName(self, uri, gen_prefix=True):
        if not isinstance(uri, URIRef):
            return None
        parts = None
        try:
            parts = self.store.compute_qname(uri, generate=gen_prefix)
        except Exception:
            pfx = self.store.store.prefix(uri)
            if pfx is not None:
                parts = (pfx, uri, '')
            else:
                return None
        prefix, namespace, local = parts
        local = local.replace('(', '\\(').replace(')', '\\)')
        if local.endswith('.'):
            return None
        prefix = self.addNamespace(prefix, namespace)
        return '%s:%s' % (prefix, local)

    def startDocument(self):
        self._started = True
        ns_list = sorted(self.namespaces.items())
        if self.base:
            self.write(self.indent() + '@base <%s> .\n' % self.base)
        for prefix, uri in ns_list:
            self.write(self.indent() + '@prefix %s: <%s> .\n' % (prefix, uri))
        if ns_list and self._spacious:
            self.write('\n')

    def endDocument(self):
        if self._spacious:
            self.write('\n')

    def statement(self, subject):
        self.subjectDone(subject)
        return self.s_squared(subject) or self.s_default(subject)

    def s_default(self, subject):
        self.write('\n' + self.indent())
        self.path(subject, SUBJECT)
        self.predicateList(subject)
        self.write(' .')
        return True

    def s_squared(self, subject):
        if self._references[subject] > 0 or not isinstance(subject, BNode):
            return False
        self.write('\n' + self.indent() + '[]')
        self.predicateList(subject)
        self.write(' .')
        return True

    def path(self, node, position, newline=False):
        if not (self.p_squared(node, position, newline) or self.p_default(node, position, newline)):
            raise Error("Cannot serialize node '%s'" % (node,))

    def p_default(self, node, position, newline=False):
        if position != SUBJECT and (not newline):
            self.write(' ')
        self.write(self.label(node, position))
        return True

    def label(self, node, position):
        if node == RDF.nil:
            return '()'
        if position is VERB and node in self.keywords:
            return self.keywords[node]
        if isinstance(node, Literal):
            return node._literal_n3(use_plain=True, qname_callback=lambda dt: self.getQName(dt, _GEN_QNAME_FOR_DT))
        else:
            node = self.relativize(node)
            return self.getQName(node, position == VERB) or node.n3()

    def p_squared(self, node, position, newline=False):
        if not isinstance(node, BNode) or node in self._serialized or self._references[node] > 1 or (position == SUBJECT):
            return False
        if not newline:
            self.write(' ')
        if self.isValidList(node):
            self.write('(')
            self.depth += 1
            self.doList(node)
            self.depth -= 1
            self.write(' )')
        else:
            self.subjectDone(node)
            self.depth += 2
            self.write('[')
            self.depth -= 1
            self.predicateList(node, newline=False)
            self.write(' ]')
            self.depth -= 1
        return True

    def isValidList(self, l_):
        """
        Checks if l is a valid RDF list, i.e. no nodes have other properties.
        """
        try:
            if self.store.value(l_, RDF.first) is None:
                return False
        except Exception:
            return False
        while l_:
            if l_ != RDF.nil and len(list(self.store.predicate_objects(l_))) != 2:
                return False
            l_ = self.store.value(l_, RDF.rest)
        return True

    def doList(self, l_):
        while l_:
            item = self.store.value(l_, RDF.first)
            if item is not None:
                self.path(item, OBJECT)
                self.subjectDone(l_)
            l_ = self.store.value(l_, RDF.rest)

    def predicateList(self, subject, newline=False):
        properties = self.buildPredicateHash(subject)
        propList = self.sortProperties(properties)
        if len(propList) == 0:
            return
        self.verb(propList[0], newline=newline)
        self.objectList(properties[propList[0]])
        for predicate in propList[1:]:
            self.write(' ;\n' + self.indent(1))
            self.verb(predicate, newline=True)
            self.objectList(properties[predicate])

    def verb(self, node, newline=False):
        self.path(node, VERB, newline)

    def objectList(self, objects):
        count = len(objects)
        if count == 0:
            return
        depthmod = count == 1 and 0 or 1
        self.depth += depthmod
        self.path(objects[0], OBJECT)
        for obj in objects[1:]:
            self.write(',\n' + self.indent(1))
            self.path(obj, OBJECT, newline=True)
        self.depth -= depthmod