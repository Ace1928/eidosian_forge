from hashlib import md5
from typing import Any, Dict, List, Optional
from langchain_core.utils import get_from_dict_or_env
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_store import GraphStore
def _get_rel_import_query(baseEntityLabel: bool) -> str:
    if baseEntityLabel:
        return f"UNWIND $data AS row MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.source}}) MERGE (target:`{BASE_ENTITY_LABEL}` {{id: row.target}}) WITH source, target, row CALL apoc.merge.relationship(source, row.type, {{}}, row.properties, target) YIELD rel RETURN distinct 'done'"
    else:
        return "UNWIND $data AS row CALL apoc.merge.node([row.source_label], {id: row.source},{}, {}) YIELD node as source CALL apoc.merge.node([row.target_label], {id: row.target},{}, {}) YIELD node as target CALL apoc.merge.relationship(source, row.type, {}, row.properties, target) YIELD rel RETURN distinct 'done'"