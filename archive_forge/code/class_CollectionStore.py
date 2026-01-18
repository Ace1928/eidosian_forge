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