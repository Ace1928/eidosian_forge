import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
from langchain_community.vectorstores import (
from langchain_community.vectorstores import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langchain.callbacks.manager import (
from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.chains.query_constructor.ir import StructuredQuery, Visitor
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.astradb import AstraDBTranslator
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.retrievers.self_query.dashvector import DashvectorTranslator
from langchain.retrievers.self_query.deeplake import DeepLakeTranslator
from langchain.retrievers.self_query.dingo import DingoDBTranslator
from langchain.retrievers.self_query.elasticsearch import ElasticsearchTranslator
from langchain.retrievers.self_query.milvus import MilvusTranslator
from langchain.retrievers.self_query.mongodb_atlas import MongoDBAtlasTranslator
from langchain.retrievers.self_query.myscale import MyScaleTranslator
from langchain.retrievers.self_query.opensearch import OpenSearchTranslator
from langchain.retrievers.self_query.pgvector import PGVectorTranslator
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from langchain.retrievers.self_query.qdrant import QdrantTranslator
from langchain.retrievers.self_query.redis import RedisTranslator
from langchain.retrievers.self_query.supabase import SupabaseVectorTranslator
from langchain.retrievers.self_query.tencentvectordb import TencentVectorDBTranslator
from langchain.retrievers.self_query.timescalevector import TimescaleVectorTranslator
from langchain.retrievers.self_query.vectara import VectaraTranslator
from langchain.retrievers.self_query.weaviate import WeaviateTranslator
def _get_builtin_translator(vectorstore: VectorStore) -> Visitor:
    """Get the translator class corresponding to the vector store class."""
    BUILTIN_TRANSLATORS: Dict[Type[VectorStore], Type[Visitor]] = {AstraDB: AstraDBTranslator, PGVector: PGVectorTranslator, Pinecone: PineconeTranslator, Chroma: ChromaTranslator, DashVector: DashvectorTranslator, Dingo: DingoDBTranslator, Weaviate: WeaviateTranslator, Vectara: VectaraTranslator, Qdrant: QdrantTranslator, MyScale: MyScaleTranslator, DeepLake: DeepLakeTranslator, ElasticsearchStoreCommunity: ElasticsearchTranslator, Milvus: MilvusTranslator, SupabaseVectorStore: SupabaseVectorTranslator, TimescaleVector: TimescaleVectorTranslator, OpenSearchVectorSearch: OpenSearchTranslator, MongoDBAtlasVectorSearch: MongoDBAtlasTranslator}
    if isinstance(vectorstore, Qdrant):
        return QdrantTranslator(metadata_key=vectorstore.metadata_payload_key)
    elif isinstance(vectorstore, MyScale):
        return MyScaleTranslator(metadata_key=vectorstore.metadata_column)
    elif isinstance(vectorstore, Redis):
        return RedisTranslator.from_vectorstore(vectorstore)
    elif isinstance(vectorstore, TencentVectorDB):
        fields = [field.name for field in vectorstore.meta_fields or [] if field.index]
        return TencentVectorDBTranslator(fields)
    elif vectorstore.__class__ in BUILTIN_TRANSLATORS:
        return BUILTIN_TRANSLATORS[vectorstore.__class__]()
    else:
        try:
            from langchain_astradb.vectorstores import AstraDBVectorStore
            if isinstance(vectorstore, AstraDBVectorStore):
                return AstraDBTranslator()
        except ImportError:
            pass
        try:
            from langchain_elasticsearch.vectorstores import ElasticsearchStore
            if isinstance(vectorstore, ElasticsearchStore):
                return ElasticsearchTranslator()
        except ImportError:
            pass
        raise ValueError(f'Self query retriever with Vector Store type {vectorstore.__class__} not supported.')