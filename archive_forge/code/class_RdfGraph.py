from __future__ import annotations
from typing import (
class RdfGraph:
    """RDFlib wrapper for graph operations.

    Modes:
    * local: Local file - can be queried and changed
    * online: Online file - can only be queried, changes can be stored locally
    * store: Triple store - can be queried and changed if update_endpoint available
    Together with a source file, the serialization should be specified.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(self, source_file: Optional[str]=None, serialization: Optional[str]='ttl', query_endpoint: Optional[str]=None, update_endpoint: Optional[str]=None, standard: Optional[str]='rdf', local_copy: Optional[str]=None, graph_kwargs: Optional[Dict]=None) -> None:
        """
        Set up the RDFlib graph

        :param source_file: either a path for a local file or a URL
        :param serialization: serialization of the input
        :param query_endpoint: SPARQL endpoint for queries, read access
        :param update_endpoint: SPARQL endpoint for UPDATE queries, write access
        :param standard: RDF, RDFS, or OWL
        :param local_copy: new local copy for storing changes
        :param graph_kwargs: Additional rdflib.Graph specific kwargs
        that will be used to initialize it,
        if query_endpoint is provided.
        """
        self.source_file = source_file
        self.serialization = serialization
        self.query_endpoint = query_endpoint
        self.update_endpoint = update_endpoint
        self.standard = standard
        self.local_copy = local_copy
        try:
            import rdflib
            from rdflib.plugins.stores import sparqlstore
        except ImportError:
            raise ValueError('Could not import rdflib python package. Please install it with `pip install rdflib`.')
        if self.standard not in (supported_standards := ('rdf', 'rdfs', 'owl')):
            raise ValueError(f'Invalid standard. Supported standards are: {supported_standards}.')
        if not source_file and (not query_endpoint) or (source_file and (query_endpoint or update_endpoint)):
            raise ValueError('Could not unambiguously initialize the graph wrapper. Specify either a file (local or online) via the source_file or a triple store via the endpoints.')
        if source_file:
            if source_file.startswith('http'):
                self.mode = 'online'
            else:
                self.mode = 'local'
                if self.local_copy is None:
                    self.local_copy = self.source_file
            self.graph = rdflib.Graph()
            self.graph.parse(source_file, format=self.serialization)
        if query_endpoint:
            self.mode = 'store'
            if not update_endpoint:
                self._store = sparqlstore.SPARQLStore()
                self._store.open(query_endpoint)
            else:
                self._store = sparqlstore.SPARQLUpdateStore()
                self._store.open((query_endpoint, update_endpoint))
            graph_kwargs = graph_kwargs or {}
            self.graph = rdflib.Graph(self._store, **graph_kwargs)
        if not len(self.graph):
            raise AssertionError('The graph is empty.')
        self.schema = ''
        self.load_schema()

    @property
    def get_schema(self) -> str:
        """
        Returns the schema of the graph database.
        """
        return self.schema

    def query(self, query: str) -> List[rdflib.query.ResultRow]:
        """
        Query the graph.
        """
        from rdflib.exceptions import ParserError
        from rdflib.query import ResultRow
        try:
            res = self.graph.query(query)
        except ParserError as e:
            raise ValueError(f'Generated SPARQL statement is invalid\n{e}')
        return [r for r in res if isinstance(r, ResultRow)]

    def update(self, query: str) -> None:
        """
        Update the graph.
        """
        from rdflib.exceptions import ParserError
        try:
            self.graph.update(query)
        except ParserError as e:
            raise ValueError(f'Generated SPARQL statement is invalid\n{e}')
        if self.local_copy:
            self.graph.serialize(destination=self.local_copy, format=self.local_copy.split('.')[-1])
        else:
            raise ValueError('No target file specified for saving the updated file.')

    @staticmethod
    def _get_local_name(iri: str) -> str:
        if '#' in iri:
            local_name = iri.split('#')[-1]
        elif '/' in iri:
            local_name = iri.split('/')[-1]
        else:
            raise ValueError(f"Unexpected IRI '{iri}', contains neither '#' nor '/'.")
        return local_name

    def _res_to_str(self, res: rdflib.query.ResultRow, var: str) -> str:
        return '<' + str(res[var]) + '> (' + self._get_local_name(res[var]) + ', ' + str(res['com']) + ')'

    def load_schema(self) -> None:
        """
        Load the graph schema information.
        """

        def _rdf_s_schema(classes: List[rdflib.query.ResultRow], relationships: List[rdflib.query.ResultRow]) -> str:
            return f'In the following, each IRI is followed by the local name and optionally its description in parentheses. \nThe RDF graph supports the following node types:\n{', '.join([self._res_to_str(r, 'cls') for r in classes])}\nThe RDF graph supports the following relationships:\n{', '.join([self._res_to_str(r, 'rel') for r in relationships])}\n'
        if self.standard == 'rdf':
            clss = self.query(cls_query_rdf)
            rels = self.query(rel_query_rdf)
            self.schema = _rdf_s_schema(clss, rels)
        elif self.standard == 'rdfs':
            clss = self.query(cls_query_rdfs)
            rels = self.query(rel_query_rdfs)
            self.schema = _rdf_s_schema(clss, rels)
        elif self.standard == 'owl':
            clss = self.query(cls_query_owl)
            ops = self.query(op_query_owl)
            dps = self.query(dp_query_owl)
            self.schema = f'In the following, each IRI is followed by the local name and optionally its description in parentheses. \nThe OWL graph supports the following node types:\n{', '.join([self._res_to_str(r, 'cls') for r in clss])}\nThe OWL graph supports the following object properties, i.e., relationships between objects:\n{', '.join([self._res_to_str(r, 'op') for r in ops])}\nThe OWL graph supports the following data properties, i.e., relationships between objects and literals:\n{', '.join([self._res_to_str(r, 'dp') for r in dps])}\n'
        else:
            raise ValueError(f"Mode '{self.standard}' is currently not supported.")