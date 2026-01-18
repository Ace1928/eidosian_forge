import logging
from typing import Any, Dict, List, Optional
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
def _result_to_document(self, outline_res: Any) -> Document:
    main_meta = {'title': outline_res['document']['title'], 'source': self.outline_instance_url + outline_res['document']['url']}
    add_meta = {'id': outline_res['document']['id'], 'ranking': outline_res['ranking'], 'collection_id': outline_res['document']['collectionId'], 'parent_document_id': outline_res['document']['parentDocumentId'], 'revision': outline_res['document']['revision'], 'created_by': outline_res['document']['createdBy']['name']} if self.load_all_available_meta else {}
    doc = Document(page_content=outline_res['document']['text'][:self.doc_content_chars_max], metadata={**main_meta, **add_meta})
    return doc