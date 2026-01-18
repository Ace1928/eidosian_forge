from __future__ import annotations
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
from langchain_core.documents import BaseDocumentTransformer, Document
def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
    text = separator.join(docs)
    if self._strip_whitespace:
        text = text.strip()
    if text == '':
        return None
    else:
        return text