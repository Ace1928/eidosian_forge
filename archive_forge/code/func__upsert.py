import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _upsert(push_doc_list: List[Dict]) -> List[str]:
    if push_doc_list is None or len(push_doc_list) == 0:
        return []
    try:
        push_request = models.PushDocumentsRequest(self.options_headers, push_doc_list)
        push_response = self.ha3_engine_client.push_documents(self.config.opt_table_name, field_name_map['id'], push_request)
        json_response = json.loads(push_response.body)
        if json_response['status'] == 'OK':
            return [push_doc['fields'][field_name_map['id']] for push_doc in push_doc_list]
        return []
    except Exception as e:
        logger.error(f'add doc to endpoint:{self.config.endpoint} instance_id:{self.config.instance_id} failed.', e)
        raise e