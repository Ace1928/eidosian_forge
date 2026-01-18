from hashlib import md5
from typing import Any, Dict, List, Optional
from langchain_core.utils import get_from_dict_or_env
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
def _get_node_import_query(baseEntityLabel: bool, include_source: bool) -> str:
    if baseEntityLabel:
        return f"{(include_docs_query if include_source else '')}UNWIND $data AS row MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.id}}) SET source += row.properties {('MERGE (d)-[:MENTIONS]->(source) ' if include_source else '')}WITH source, row CALL apoc.create.addLabels( source, [row.type] ) YIELD node RETURN distinct 'done' AS result"
    else:
        return f"{(include_docs_query if include_source else '')}UNWIND $data AS row CALL apoc.merge.node([row.type], {{id: row.id}}, row.properties, {{}}) YIELD node {('MERGE (d)-[:MENTIONS]->(node) ' if include_source else '')}RETURN distinct 'done' AS result"