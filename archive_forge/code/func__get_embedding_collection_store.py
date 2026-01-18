from __future__ import annotations
import contextlib
import enum
import json
import logging
import uuid
from typing import (
import numpy as np
import sqlalchemy
from langchain_core._api import deprecated, warn_deprecated
from sqlalchemy import SQLColumnExpression, delete, func
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import Session, relationship
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _get_embedding_collection_store(vector_dimension: Optional[int]=None, *, use_jsonb: bool=True) -> Any:
    global _classes
    if _classes is not None:
        return _classes
    from pgvector.sqlalchemy import Vector

    class CollectionStore(BaseModel):
        """Collection store."""
        __tablename__ = 'langchain_pg_collection'
        name = sqlalchemy.Column(sqlalchemy.String)
        cmetadata = sqlalchemy.Column(JSON)
        embeddings = relationship('EmbeddingStore', back_populates='collection', passive_deletes=True)

        @classmethod
        def get_by_name(cls, session: Session, name: str) -> Optional['CollectionStore']:
            return session.query(cls).filter(cls.name == name).first()

        @classmethod
        def get_or_create(cls, session: Session, name: str, cmetadata: Optional[dict]=None) -> Tuple['CollectionStore', bool]:
            """
            Get or create a collection.
            Returns [Collection, bool] where the bool is True if the collection was created.
            """
            created = False
            collection = cls.get_by_name(session, name)
            if collection:
                return (collection, created)
            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            session.commit()
            created = True
            return (collection, created)
    if use_jsonb:

        class EmbeddingStore(BaseModel):
            """Embedding store."""
            __tablename__ = 'langchain_pg_embedding'
            collection_id = sqlalchemy.Column(UUID(as_uuid=True), sqlalchemy.ForeignKey(f'{CollectionStore.__tablename__}.uuid', ondelete='CASCADE'))
            collection = relationship(CollectionStore, back_populates='embeddings')
            embedding: Vector = sqlalchemy.Column(Vector(vector_dimension))
            document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
            cmetadata = sqlalchemy.Column(JSONB, nullable=True)
            custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)
            __table_args__ = (sqlalchemy.Index('ix_cmetadata_gin', 'cmetadata', postgresql_using='gin', postgresql_ops={'cmetadata': 'jsonb_path_ops'}),)
    else:

        class EmbeddingStore(BaseModel):
            """Embedding store."""
            __tablename__ = 'langchain_pg_embedding'
            collection_id = sqlalchemy.Column(UUID(as_uuid=True), sqlalchemy.ForeignKey(f'{CollectionStore.__tablename__}.uuid', ondelete='CASCADE'))
            collection = relationship(CollectionStore, back_populates='embeddings')
            embedding: Vector = sqlalchemy.Column(Vector(vector_dimension))
            document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
            cmetadata = sqlalchemy.Column(JSON, nullable=True)
            custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    _classes = (EmbeddingStore, CollectionStore)
    return _classes