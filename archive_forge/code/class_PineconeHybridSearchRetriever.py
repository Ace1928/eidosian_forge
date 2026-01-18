import hashlib
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.retrievers import BaseRetriever
class PineconeHybridSearchRetriever(BaseRetriever):
    """`Pinecone Hybrid Search` retriever."""
    embeddings: Embeddings
    'Embeddings model to use.'
    'description'
    sparse_encoder: Any
    'Sparse encoder to use.'
    index: Any
    'Pinecone index to use.'
    top_k: int = 4
    'Number of documents to return.'
    alpha: float = 0.5
    'Alpha value for hybrid search.'
    namespace: Optional[str] = None
    'Namespace value for index partition.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_texts(self, texts: List[str], ids: Optional[List[str]]=None, metadatas: Optional[List[dict]]=None, namespace: Optional[str]=None) -> None:
        create_index(texts, self.index, self.embeddings, self.sparse_encoder, ids=ids, metadatas=metadatas, namespace=namespace)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from pinecone_text.hybrid import hybrid_convex_scale
            from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder
        except ImportError:
            raise ImportError('Could not import pinecone_text python package. Please install it with `pip install pinecone_text`.')
        return values

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        from pinecone_text.hybrid import hybrid_convex_scale
        sparse_vec = self.sparse_encoder.encode_queries(query)
        dense_vec = self.embeddings.embed_query(query)
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec['values'] = [float(s1) for s1 in sparse_vec['values']]
        result = self.index.query(vector=dense_vec, sparse_vector=sparse_vec, top_k=self.top_k, include_metadata=True, namespace=self.namespace)
        final_result = []
        for res in result['matches']:
            context = res['metadata'].pop('context')
            final_result.append(Document(page_content=context, metadata=res['metadata']))
        return final_result