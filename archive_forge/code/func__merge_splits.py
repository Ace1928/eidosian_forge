from __future__ import annotations
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
from langchain_core.documents import BaseDocumentTransformer, Document
def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
    separator_len = self._length_function(separator)
    docs = []
    current_doc: List[str] = []
    total = 0
    for d in splits:
        _len = self._length_function(d)
        if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
            if total > self._chunk_size:
                logger.warning(f'Created a chunk of size {total}, which is longer than the specified {self._chunk_size}')
            if len(current_doc) > 0:
                doc = self._join_docs(current_doc, separator)
                if doc is not None:
                    docs.append(doc)
                while total > self._chunk_overlap or (total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size and total > 0):
                    total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                    current_doc = current_doc[1:]
        current_doc.append(d)
        total += _len + (separator_len if len(current_doc) > 1 else 0)
    doc = self._join_docs(current_doc, separator)
    if doc is not None:
        docs.append(doc)
    return docs