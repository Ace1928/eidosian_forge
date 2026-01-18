import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
class SPARQLStore(SPARQLConnector, Store):
    """An RDFLib store around a SPARQL endpoint

    This is context-aware and should work as expected
    when a context is specified.

    For ConjunctiveGraphs, reading is done from the "default graph". Exactly
    what this means depends on your endpoint, because SPARQL does not offer a
    simple way to query the union of all graphs as it would be expected for a
    ConjuntiveGraph. This is why we recommend using Dataset instead, which is
    motivated by the SPARQL 1.1.

    Fuseki/TDB has a flag for specifying that the default graph
    is the union of all graphs (``tdb:unionDefaultGraph`` in the Fuseki config).

    .. warning:: By default the SPARQL Store does not support blank-nodes!

                 As blank-nodes act as variables in SPARQL queries,
                 there is no way to query for a particular blank node without
                 using non-standard SPARQL extensions.

                 See http://www.w3.org/TR/sparql11-query/#BGPsparqlBNodes

    You can make use of such extensions through the ``node_to_sparql``
    argument. For example if you want to transform BNode('0001') into
    "<bnode:b0001>", you can use a function like this:

    >>> def my_bnode_ext(node):
    ...    if isinstance(node, BNode):
    ...        return '<bnode:b%s>' % node
    ...    return _node_to_sparql(node)
    >>> store = SPARQLStore('http://dbpedia.org/sparql',
    ...                     node_to_sparql=my_bnode_ext)

    You can request a particular result serialization with the
    ``returnFormat`` parameter. This is a string that must have a
    matching plugin registered. Built in is support for ``xml``,
    ``json``, ``csv``, ``tsv`` and ``application/rdf+xml``.

    The underlying SPARQLConnector uses the urllib library.
    Any extra kwargs passed to the SPARQLStore connector are passed to
    urllib when doing HTTP calls. I.e. you have full control of
    cookies/auth/headers.

    Form example:

    >>> store = SPARQLStore('...my endpoint ...', auth=('user','pass'))

    will use HTTP basic auth.

    """
    formula_aware = False
    transaction_aware = False
    graph_aware = True
    regex_matching = NATIVE_REGEX

    def __init__(self, query_endpoint: Optional[str]=None, sparql11: bool=True, context_aware: bool=True, node_to_sparql: _NodeToSparql=_node_to_sparql, returnFormat: str='xml', auth: Optional[Tuple[str, str]]=None, **sparqlconnector_kwargs):
        super(SPARQLStore, self).__init__(query_endpoint=query_endpoint, returnFormat=returnFormat, auth=auth, **sparqlconnector_kwargs)
        self.node_to_sparql = node_to_sparql
        self.nsBindings: Dict[str, Any] = {}
        self.sparql11 = sparql11
        self.context_aware = context_aware
        self.graph_aware = context_aware
        self._queries = 0

    def open(self, configuration: str, create: bool=False) -> Optional[int]:
        """This method is included so that calls to this Store via Graph, e.g. Graph("SPARQLStore"),
        can set the required parameters
        """
        if type(configuration) == str:
            self.query_endpoint = configuration
        else:
            raise Exception('configuration must be a string (a single query endpoint URI)')

    def create(self, configuration: str) -> None:
        raise TypeError('The SPARQL Store is read only. Try SPARQLUpdateStore for read/write.')

    def destroy(self, configuration: str) -> None:
        raise TypeError('The SPARQL store is read only')

    def commit(self) -> None:
        raise TypeError('The SPARQL store is read only')

    def rollback(self) -> None:
        raise TypeError('The SPARQL store is read only')

    def add(self, _: '_TripleType', context: '_ContextType'=None, quoted: bool=False) -> None:
        raise TypeError('The SPARQL store is read only')

    def addN(self, quads: Iterable['_QuadType']) -> None:
        raise TypeError('The SPARQL store is read only')

    def remove(self, _: '_TriplePatternType', context: Optional['_ContextType']) -> None:
        raise TypeError('The SPARQL store is read only')

    def update(self, query: Union['Update', str], initNs: Dict[str, Any]={}, initBindings: Dict['str', 'Identifier']={}, queryGraph: 'Identifier'=None, DEBUG: bool=False) -> None:
        raise TypeError('The SPARQL store is read only')

    def _query(self, *args: Any, **kwargs: Any) -> 'Result':
        self._queries += 1
        return super(SPARQLStore, self).query(*args, **kwargs)

    def _inject_prefixes(self, query: str, extra_bindings: Mapping[str, Any]) -> str:
        bindings = set(list(self.nsBindings.items()) + list(extra_bindings.items()))
        if not bindings:
            return query
        return '\n'.join(['\n'.join(['PREFIX %s: <%s>' % (k, v) for k, v in bindings]), '', query])

    def query(self, query: Union['Query', str], initNs: Optional[Mapping[str, Any]]=None, initBindings: Optional[Mapping['str', 'Identifier']]=None, queryGraph: Optional['str']=None, DEBUG: bool=False) -> 'Result':
        self.debug = DEBUG
        assert isinstance(query, str)
        if initNs is not None and len(initNs) > 0:
            query = self._inject_prefixes(query, initNs)
        if initBindings:
            if not self.sparql11:
                raise Exception('initBindings not supported for SPARQL 1.0 Endpoints.')
            v = list(initBindings)
            query += '\nVALUES ( %s )\n{ ( %s ) }\n' % (' '.join(('?' + str(x) for x in v)), ' '.join((self.node_to_sparql(initBindings[x]) for x in v)))
        return self._query(query, default_graph=queryGraph if self._is_contextual(queryGraph) else None)

    def triples(self, spo: '_TriplePatternType', context: Optional['_ContextType']=None) -> Iterator[Tuple['_TripleType', None]]:
        """
        - tuple **(s, o, p)**
          the triple used as filter for the SPARQL select.
          (None, None, None) means anything.
        - context **context**
          the graph effectively calling this method.

        Returns a tuple of triples executing essentially a SPARQL like
        SELECT ?subj ?pred ?obj WHERE { ?subj ?pred ?obj }

        **context** may include three parameter
        to refine the underlying query:

        * LIMIT: an integer to limit the number of results
        * OFFSET: an integer to enable paging of results
        * ORDERBY: an instance of Variable('s'), Variable('o') or Variable('p') or, by default, the first 'None' from the given triple

        .. warning::

            - Using LIMIT or OFFSET automatically include ORDERBY otherwise this is
              because the results are retrieved in a not deterministic way (depends on
              the walking path on the graph)
            - Using OFFSET without defining LIMIT will discard the first OFFSET - 1 results

        .. code-block:: python

            a_graph.LIMIT = limit
            a_graph.OFFSET = offset
            triple_generator = a_graph.triples(mytriple):
            # do something
            # Removes LIMIT and OFFSET if not required for the next triple() calls
            del a_graph.LIMIT
            del a_graph.OFFSET
        """
        s, p, o = spo
        vars = []
        if not s:
            s = Variable('s')
            vars.append(s)
        if not p:
            p = Variable('p')
            vars.append(p)
        if not o:
            o = Variable('o')
            vars.append(o)
        if vars:
            v = ' '.join([term.n3() for term in vars])
            verb = 'SELECT %s ' % v
        else:
            verb = 'ASK'
        nts = self.node_to_sparql
        query = '%s { %s %s %s }' % (verb, nts(s), nts(p), nts(o))
        if hasattr(context, LIMIT) or hasattr(context, OFFSET) or hasattr(context, ORDERBY):
            var = None
            if isinstance(s, Variable):
                var = s
            elif isinstance(p, Variable):
                var = p
            elif isinstance(o, Variable):
                var = o
            elif hasattr(context, ORDERBY) and isinstance(getattr(context, ORDERBY), Variable):
                var = getattr(context, ORDERBY)
            query = query + ' %s %s' % (ORDERBY, var.n3())
        try:
            query = query + ' LIMIT %s' % int(getattr(context, LIMIT))
        except (ValueError, TypeError, AttributeError):
            pass
        try:
            query = query + ' OFFSET %s' % int(getattr(context, OFFSET))
        except (ValueError, TypeError, AttributeError):
            pass
        result = self._query(query, default_graph=context.identifier if self._is_contextual(context) else None)
        if vars:
            if type(result) == tuple:
                if result[0] == 401:
                    raise ValueError('It looks like you need to authenticate with this SPARQL Store. HTTP unauthorized')
            for row in result:
                if TYPE_CHECKING:
                    assert isinstance(row, ResultRow)
                yield ((row.get(s, s), row.get(p, p), row.get(o, o)), None)
        elif result.askAnswer:
            yield ((s, p, o), None)

    def triples_choices(self, _: Tuple[Union['_SubjectType', List['_SubjectType']], Union['_PredicateType', List['_PredicateType']], Union['_ObjectType', List['_ObjectType']]], context: Optional['_ContextType']=None) -> Generator[Tuple[Tuple['_SubjectType', '_PredicateType', '_ObjectType'], Iterator[Optional['_ContextType']]], None, None]:
        """
        A variant of triples that can take a list of terms instead of a
        single term in any slot.  Stores can implement this to optimize
        the response time from the import default 'fallback' implementation,
        which will iterate over each term in the list and dispatch to
        triples.
        """
        raise NotImplementedError('Triples choices currently not supported')

    def __len__(self, context: Optional['_ContextType']=None) -> int:
        if not self.sparql11:
            raise NotImplementedError('For performance reasons, this is not' + 'supported for sparql1.0 endpoints')
        else:
            q = 'SELECT (count(*) as ?c) WHERE {?s ?p ?o .}'
            result = self._query(q, default_graph=context.identifier if self._is_contextual(context) else None)
            return int(next(iter(result)).c)

    def contexts(self, triple: Optional['_TripleType']=None) -> Generator['_ContextIdentifierType', None, None]:
        """
        Iterates over results to "SELECT ?NAME { GRAPH ?NAME { ?s ?p ?o } }"
        or "SELECT ?NAME { GRAPH ?NAME {} }" if triple is `None`.

        Returns instances of this store with the SPARQL wrapper
        object updated via addNamedGraph(?NAME).

        This causes a named-graph-uri key / value  pair to be sent over
        the protocol.

        Please note that some SPARQL endpoints are not able to find empty named
        graphs.
        """
        if triple:
            nts = self.node_to_sparql
            s, p, o = triple
            params = (nts(s if s else Variable('s')), nts(p if p else Variable('p')), nts(o if o else Variable('o')))
            q = 'SELECT ?name WHERE { GRAPH ?name { %s %s %s }}' % params
        else:
            q = 'SELECT ?name WHERE { GRAPH ?name {} }'
        result = self._query(q)
        return (row.name for row in result)

    def bind(self, prefix: str, namespace: 'URIRef', override: bool=True) -> None:
        bound_prefix = self.prefix(namespace)
        if override and bound_prefix:
            del self.nsBindings[bound_prefix]
        self.nsBindings[prefix] = namespace

    def prefix(self, namespace: 'URIRef') -> Optional['str']:
        """ """
        return dict([(v, k) for k, v in self.nsBindings.items()]).get(namespace)

    def namespace(self, prefix: str) -> Optional['URIRef']:
        return self.nsBindings.get(prefix)

    def namespaces(self) -> Iterator[Tuple[str, 'URIRef']]:
        for prefix, ns in self.nsBindings.items():
            yield (prefix, ns)

    def add_graph(self, graph: 'Graph') -> None:
        raise TypeError('The SPARQL store is read only')

    def remove_graph(self, graph: 'Graph') -> None:
        raise TypeError('The SPARQL store is read only')

    @overload
    def _is_contextual(self, graph: None) -> 'te.Literal[False]':
        ...

    @overload
    def _is_contextual(self, graph: Optional[Union['Graph', 'str']]) -> bool:
        ...

    def _is_contextual(self, graph: Optional[Union['Graph', 'str']]) -> bool:
        """Returns `True` if the "GRAPH" keyword must appear
        in the final SPARQL query sent to the endpoint.
        """
        if not self.context_aware or graph is None:
            return False
        if isinstance(graph, str):
            return graph != '__UNION__'
        else:
            return graph.identifier != DATASET_DEFAULT_GRAPH_ID

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