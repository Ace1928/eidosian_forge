from __future__ import annotations
import os
from typing import (
def _load_ontology_schema_with_query(self, query: str):
    """
        Execute the query for collecting the ontology schema statements
        """
    from rdflib.exceptions import ParserError
    try:
        results = self.graph.query(query)
    except ParserError as e:
        raise ValueError(f'Generated SPARQL statement is invalid\n{e}')
    return results.graph