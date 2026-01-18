from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_community.tools.azure_cognitive_services.utils import (
def _document_analysis(self, document_path: str) -> Dict:
    document_src_type = detect_file_src_type(document_path)
    if document_src_type == 'local':
        with open(document_path, 'rb') as document:
            poller = self.doc_analysis_client.begin_analyze_document('prebuilt-document', document)
    elif document_src_type == 'remote':
        poller = self.doc_analysis_client.begin_analyze_document_from_url('prebuilt-document', document_path)
    else:
        raise ValueError(f'Invalid document path: {document_path}')
    result = poller.result()
    res_dict = {}
    if result.content is not None:
        res_dict['content'] = result.content
    if result.tables is not None:
        res_dict['tables'] = self._parse_tables(result.tables)
    if result.key_value_pairs is not None:
        res_dict['key_value_pairs'] = self._parse_kv_pairs(result.key_value_pairs)
    return res_dict