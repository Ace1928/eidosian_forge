from __future__ import annotations
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
def _initialize_table(self) -> Any:
    """Validates or creates the BigQuery table."""
    from google.cloud import bigquery
    table_ref = bigquery.TableReference.from_string(self._full_table_id)
    table = self.bq_client.create_table(table_ref, exists_ok=True)
    changed_schema = False
    schema = table.schema.copy()
    columns = {c.name: c for c in schema}
    if self.doc_id_field not in columns:
        changed_schema = True
        schema.append(bigquery.SchemaField(name=self.doc_id_field, field_type='STRING'))
    elif columns[self.doc_id_field].field_type != 'STRING' or columns[self.doc_id_field].mode == 'REPEATED':
        raise ValueError(f'Column {self.doc_id_field} must be of STRING type')
    if self.metadata_field not in columns:
        changed_schema = True
        schema.append(bigquery.SchemaField(name=self.metadata_field, field_type='JSON'))
    elif columns[self.metadata_field].field_type not in ['JSON', 'STRING'] or columns[self.metadata_field].mode == 'REPEATED':
        raise ValueError(f'Column {self.metadata_field} must be of STRING or JSON type')
    if self.content_field not in columns:
        changed_schema = True
        schema.append(bigquery.SchemaField(name=self.content_field, field_type='STRING'))
    elif columns[self.content_field].field_type != 'STRING' or columns[self.content_field].mode == 'REPEATED':
        raise ValueError(f'Column {self.content_field} must be of STRING type')
    if self.text_embedding_field not in columns:
        changed_schema = True
        schema.append(bigquery.SchemaField(name=self.text_embedding_field, field_type='FLOAT64', mode='REPEATED'))
    elif columns[self.text_embedding_field].field_type not in ('FLOAT', 'FLOAT64') or columns[self.text_embedding_field].mode != 'REPEATED':
        raise ValueError(f'Column {self.text_embedding_field} must be of ARRAY<FLOAT64> type')
    if changed_schema:
        self._logger.debug('Updated table `%s` schema.', self.full_table_id)
        table.schema = schema
        table = self.bq_client.update_table(table, fields=['schema'])
    return table