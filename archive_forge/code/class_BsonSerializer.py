import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Type
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class BsonSerializer(BaseSerializer):
    """Serialize data in Binary JSON using the `bson` python package."""

    def __init__(self, persist_path: str) -> None:
        super().__init__(persist_path)
        self.bson = guard_import('bson')

    @classmethod
    def extension(cls) -> str:
        return 'bson'

    def save(self, data: Any) -> None:
        with open(self.persist_path, 'wb') as fp:
            fp.write(self.bson.dumps(data))

    def load(self) -> Any:
        with open(self.persist_path, 'rb') as fp:
            return self.bson.loads(fp.read())