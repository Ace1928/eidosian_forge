from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
class ArceeDocumentAdapter:
    """Adapter for Arcee documents"""

    @classmethod
    def adapt(cls, arcee_document: ArceeDocument) -> Document:
        """Adapts an `ArceeDocument` to a langchain's `Document` object."""
        return Document(page_content=arcee_document.source.document, metadata={'name': arcee_document.source.name, 'source_id': arcee_document.source.id, 'index': arcee_document.index, 'id': arcee_document.id, 'score': arcee_document.score})