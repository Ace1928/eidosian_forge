import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
class SPARQLUpdateStore(SPARQLStore):
    """A store using SPARQL queries for reading and SPARQL Update for changes.

    This can be context-aware, if so, any changes will be to the given named
    graph only.

    In favor of the SPARQL 1.1 motivated Dataset, we advise against using this
    with ConjunctiveGraphs, as it reads and writes from and to the
    "default graph". Exactly what this means depends on the endpoint and can
    result in confusion.

    For Graph objects, everything works as expected.

    See the :class:`SPARQLStore` base class for more information.

    """
    where_pattern = re.compile('(?P<where>WHERE\\s*\\{)', re.IGNORECASE)
    STRING_LITERAL1 = "'([^'\\\\]|\\\\.)*'"
    STRING_LITERAL2 = '"([^"\\\\]|\\\\.)*"'
    STRING_LITERAL_LONG1 = "'''(('|'')?([^'\\\\]|\\\\.))*'''"
    STRING_LITERAL_LONG2 = '"""(("|"")?([^"\\\\]|\\\\.))*"""'
    String = '(%s)|(%s)|(%s)|(%s)' % (STRING_LITERAL1, STRING_LITERAL2, STRING_LITERAL_LONG1, STRING_LITERAL_LONG2)
    IRIREF = '<([^<>"{}|^`\\]\\\\[\\x00-\\x20])*>'
    COMMENT = '#[^\\x0D\\x0A]*([\\x0D\\x0A]|\\Z)'
    BLOCK_START = '{'
    BLOCK_END = '}'
    ESCAPED = '\\\\.'
    BlockContent = '(%s)|(%s)|(%s)|(%s)' % (String, IRIREF, COMMENT, ESCAPED)
    BlockFinding = '(?P<block_start>%s)|(?P<block_end>%s)|(?P<block_content>%s)' % (BLOCK_START, BLOCK_END, BlockContent)
    BLOCK_FINDING_PATTERN = re.compile(BlockFinding)

    def __init__(self, query_endpoint: Optional[str]=None, update_endpoint: Optional[str]=None, sparql11: bool=True, context_aware: bool=True, postAsEncoded: bool=True, autocommit: bool=True, dirty_reads: bool=False, **kwds):
        """
        :param autocommit if set, the store will commit after every
        writing operations. If False, we only make queries on the
        server once commit is called.

        :param dirty_reads if set, we do not commit before reading. So you
        cannot read what you wrote before manually calling commit.

        """
        SPARQLStore.__init__(self, query_endpoint, sparql11, context_aware, update_endpoint=update_endpoint, **kwds)
        self.postAsEncoded = postAsEncoded
        self.autocommit = autocommit
        self.dirty_reads = dirty_reads
        self._edits: Optional[List[str]] = None
        self._updates = 0

    def open(self, configuration: Union[str, Tuple[str, str]], create: bool=False) -> None:
        """
        This method is included so that calls to this Store via Graph, e.g.
        Graph("SPARQLStore"), can set the required parameters
        """
        if type(configuration) == str:
            self.query_endpoint = configuration
        elif type(configuration) == tuple:
            self.query_endpoint = configuration[0]
            self.update_endpoint = configuration[1]
        else:
            raise Exception('configuration must be either a string (a single query endpoint URI) or a tuple (a query/update endpoint URI pair)')

    def query(self, *args: Any, **kwargs: Any) -> 'Result':
        if not self.autocommit and (not self.dirty_reads):
            self.commit()
        return SPARQLStore.query(self, *args, **kwargs)

    def triples(self, *args: Any, **kwargs: Any) -> Iterator[Tuple['_TripleType', None]]:
        if not self.autocommit and (not self.dirty_reads):
            self.commit()
        return SPARQLStore.triples(self, *args, **kwargs)

    def contexts(self, *args: Any, **kwargs: Any) -> Generator['_ContextIdentifierType', None, None]:
        if not self.autocommit and (not self.dirty_reads):
            self.commit()
        return SPARQLStore.contexts(self, *args, **kwargs)

    def __len__(self, *args: Any, **kwargs: Any) -> int:
        if not self.autocommit and (not self.dirty_reads):
            self.commit()
        return SPARQLStore.__len__(self, *args, **kwargs)

    def open(self, configuration: Union[str, Tuple[str, str]], create: bool=False) -> None:
        """
        sets the endpoint URLs for this SPARQLStore

        :param configuration: either a tuple of (query_endpoint, update_endpoint),
            or a string with the endpoint which is configured as query and update endpoint
        :param create: if True an exception is thrown.
        """
        if create:
            raise Exception('Cannot create a SPARQL Endpoint')
        if isinstance(configuration, tuple):
            self.query_endpoint = configuration[0]
            if len(configuration) > 1:
                self.update_endpoint = configuration[1]
        else:
            self.query_endpoint = configuration
            self.update_endpoint = configuration

    def _transaction(self) -> List[str]:
        if self._edits is None:
            self._edits = []
        return self._edits

    def commit(self) -> None:
        """add(), addN(), and remove() are transactional to reduce overhead of many small edits.
        Read and update() calls will automatically commit any outstanding edits.
        This should behave as expected most of the time, except that alternating writes
        and reads can degenerate to the original call-per-triple situation that originally existed.
        """
        if self._edits and len(self._edits) > 0:
            self._update('\n;\n'.join(self._edits))
            self._edits = None

    def rollback(self) -> None:
        self._edits = None

    def add(self, spo: '_TripleType', context: Optional['_ContextType']=None, quoted: bool=False) -> None:
        """Add a triple to the store of triples."""
        if not self.update_endpoint:
            raise Exception('UpdateEndpoint is not set')
        assert not quoted
        subject, predicate, obj = spo
        nts = self.node_to_sparql
        triple = '%s %s %s .' % (nts(subject), nts(predicate), nts(obj))
        if self._is_contextual(context):
            if TYPE_CHECKING:
                assert context is not None
            q = 'INSERT DATA { GRAPH %s { %s } }' % (nts(context.identifier), triple)
        else:
            q = 'INSERT DATA { %s }' % triple
        self._transaction().append(q)
        if self.autocommit:
            self.commit()

    def addN(self, quads: Iterable['_QuadType']) -> None:
        """Add a list of quads to the store."""
        if not self.update_endpoint:
            raise Exception("UpdateEndpoint is not set - call 'open'")
        contexts = collections.defaultdict(list)
        for subject, predicate, obj, context in quads:
            contexts[context].append((subject, predicate, obj))
        data: List[str] = []
        nts = self.node_to_sparql
        for context in contexts:
            triples = ['%s %s %s .' % (nts(subject), nts(predicate), nts(obj)) for subject, predicate, obj in contexts[context]]
            data.append('INSERT DATA { GRAPH %s { %s } }\n' % (nts(context.identifier), '\n'.join(triples)))
        self._transaction().extend(data)
        if self.autocommit:
            self.commit()

    def remove(self, spo: '_TriplePatternType', context: Optional['_ContextType']) -> None:
        """Remove a triple from the store"""
        if not self.update_endpoint:
            raise Exception("UpdateEndpoint is not set - call 'open'")
        subject, predicate, obj = spo
        if not subject:
            subject = Variable('S')
        if not predicate:
            predicate = Variable('P')
        if not obj:
            obj = Variable('O')
        nts = self.node_to_sparql
        triple = '%s %s %s .' % (nts(subject), nts(predicate), nts(obj))
        if self._is_contextual(context):
            if TYPE_CHECKING:
                assert context is not None
            cid = nts(context.identifier)
            q = 'WITH %(graph)s DELETE { %(triple)s } WHERE { %(triple)s }' % {'graph': cid, 'triple': triple}
        else:
            q = 'DELETE { %s } WHERE { %s } ' % (triple, triple)
        self._transaction().append(q)
        if self.autocommit:
            self.commit()

    def setTimeout(self, timeout) -> None:
        self._timeout = int(timeout)

    def _update(self, update):
        self._updates += 1
        SPARQLConnector.update(self, update)

    def update(self, query: Union['Update', str], initNs: Dict[str, Any]={}, initBindings: Dict['str', 'Identifier']={}, queryGraph: Optional[str]=None, DEBUG: bool=False):
        """
        Perform a SPARQL Update Query against the endpoint,
        INSERT, LOAD, DELETE etc.
        Setting initNs adds PREFIX declarations to the beginning of
        the update. Setting initBindings adds inline VALUEs to the
        beginning of every WHERE clause. By the SPARQL grammar, all
        operations that support variables (namely INSERT and DELETE)
        require a WHERE clause.
        Important: initBindings fails if the update contains the
        substring 'WHERE {' which does not denote a WHERE clause, e.g.
        if it is part of a literal.

        .. admonition:: Context-aware query rewriting

            - **When:**  If context-awareness is enabled and the graph is not the default graph of the store.
            - **Why:** To ensure consistency with the :class:`~rdflib.plugins.stores.memory.Memory` store.
              The graph must accept "local" SPARQL requests (requests with no GRAPH keyword)
              as if it was the default graph.
            - **What is done:** These "local" queries are rewritten by this store.
              The content of each block of a SPARQL Update operation is wrapped in a GRAPH block
              except if the block is empty.
              This basically causes INSERT, INSERT DATA, DELETE, DELETE DATA and WHERE to operate
              only on the context.
            - **Example:** ``"INSERT DATA { <urn:michel> <urn:likes> <urn:pizza> }"`` is converted into
              ``"INSERT DATA { GRAPH <urn:graph> { <urn:michel> <urn:likes> <urn:pizza> } }"``.
            - **Warning:** Queries are presumed to be "local" but this assumption is **not checked**.
              For instance, if the query already contains GRAPH blocks, the latter will be wrapped in new GRAPH blocks.
            - **Warning:** A simplified grammar is used that should tolerate
              extensions of the SPARQL grammar. Still, the process may fail in
              uncommon situations and produce invalid output.

        """
        if not self.update_endpoint:
            raise Exception('Update endpoint is not set!')
        self.debug = DEBUG
        assert isinstance(query, str)
        query = self._inject_prefixes(query, initNs)
        if self._is_contextual(queryGraph):
            if TYPE_CHECKING:
                assert queryGraph is not None
            query = self._insert_named_graph(query, queryGraph)
        if initBindings:
            v = list(initBindings)
            values = '\nVALUES ( %s )\n{ ( %s ) }\n' % (' '.join(('?' + str(x) for x in v)), ' '.join((self.node_to_sparql(initBindings[x]) for x in v)))
            query = self.where_pattern.sub('WHERE { ' + values, query)
        self._transaction().append(query)
        if self.autocommit:
            self.commit()

    def _insert_named_graph(self, query: str, query_graph: str) -> str:
        """
        Inserts GRAPH <query_graph> {} into blocks of SPARQL Update operations

        For instance,  "INSERT DATA { <urn:michel> <urn:likes> <urn:pizza> }"
        is converted into
        "INSERT DATA { GRAPH <urn:graph> { <urn:michel> <urn:likes> <urn:pizza> } }"
        """
        if isinstance(query_graph, Node):
            query_graph = self.node_to_sparql(query_graph)
        else:
            query_graph = '<%s>' % query_graph
        graph_block_open = ' GRAPH %s {' % query_graph
        graph_block_close = '} '
        level = 0
        modified_query = []
        pos = 0
        for match in self.BLOCK_FINDING_PATTERN.finditer(query):
            if match.group('block_start') is not None:
                level += 1
                if level == 1:
                    modified_query.append(query[pos:match.end()])
                    modified_query.append(graph_block_open)
                    pos = match.end()
            elif match.group('block_end') is not None:
                if level == 1:
                    since_previous_pos = query[pos:match.start()]
                    if modified_query[-1] is graph_block_open and (since_previous_pos == '' or since_previous_pos.isspace()):
                        modified_query.pop()
                        modified_query.append(since_previous_pos)
                    else:
                        modified_query.append(since_previous_pos)
                        modified_query.append(graph_block_close)
                    pos = match.start()
                level -= 1
        modified_query.append(query[pos:])
        return ''.join(modified_query)

    def add_graph(self, graph: 'Graph') -> None:
        if not self.graph_aware:
            Store.add_graph(self, graph)
        elif graph.identifier != DATASET_DEFAULT_GRAPH_ID:
            self.update('CREATE GRAPH %s' % self.node_to_sparql(graph.identifier))

    def remove_graph(self, graph: 'Graph') -> None:
        if not self.graph_aware:
            Store.remove_graph(self, graph)
        elif graph.identifier == DATASET_DEFAULT_GRAPH_ID:
            self.update('DROP DEFAULT')
        else:
            self.update('DROP GRAPH %s' % self.node_to_sparql(graph.identifier))

    def subjects(self, predicate: Optional['_PredicateType']=None, object: Optional['_ObjectType']=None) -> Generator['_SubjectType', None, None]:
        """A generator of subjects with the given predicate and object"""
        for t, c in self.triples((None, predicate, object)):
            yield t[0]

    def predicates(self, subject: Optional['_SubjectType']=None, object: Optional['_ObjectType']=None) -> Generator['_PredicateType', None, None]:
        """A generator of predicates with the given subject and object"""
        for t, c in self.triples((subject, None, object)):
            yield t[1]

    def objects(self, subject: Optional['_SubjectType']=None, predicate: Optional['_PredicateType']=None) -> Generator['_ObjectType', None, None]:
        """A generator of objects with the given subject and predicate"""
        for t, c in self.triples((subject, predicate, None)):
            yield t[2]

    def subject_predicates(self, object: Optional['_ObjectType']=None) -> Generator[Tuple['_SubjectType', '_PredicateType'], None, None]:
        """A generator of (subject, predicate) tuples for the given object"""
        for t, c in self.triples((None, None, object)):
            yield (t[0], t[1])

    def subject_objects(self, predicate: Optional['_PredicateType']=None) -> Generator[Tuple['_SubjectType', '_ObjectType'], None, None]:
        """A generator of (subject, object) tuples for the given predicate"""
        for t, c in self.triples((None, predicate, None)):
            yield (t[0], t[2])

    def predicate_objects(self, subject: Optional['_SubjectType']=None) -> Generator[Tuple['_PredicateType', '_ObjectType'], None, None]:
        """A generator of (predicate, object) tuples for the given subject"""
        for t, c in self.triples((subject, None, None)):
            yield (t[1], t[2])