import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
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