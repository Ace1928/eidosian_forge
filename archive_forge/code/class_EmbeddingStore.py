from __future__ import annotations
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import sqlalchemy
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Session, relationship
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
class EmbeddingStore(BaseModel):
    """Embedding store."""
    __tablename__ = 'langchain_pg_embedding'
    collection_id = sqlalchemy.Column(UUID(as_uuid=True), sqlalchemy.ForeignKey(f'{CollectionStore.__tablename__}.uuid', ondelete='CASCADE'))
    collection = relationship(CollectionStore, back_populates='embeddings')
    embedding = sqlalchemy.Column(sqlalchemy.ARRAY(sqlalchemy.REAL))
    document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    cmetadata = sqlalchemy.Column(JSON, nullable=True)
    custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)