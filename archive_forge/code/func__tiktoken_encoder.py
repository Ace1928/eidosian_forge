from __future__ import annotations
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
from langchain_core.documents import BaseDocumentTransformer, Document
def _tiktoken_encoder(text: str) -> int:
    return len(enc.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special))