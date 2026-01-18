from typing import Any, Callable, List, Sequence
import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utils.math import cosine_similarity
class EmbeddingsClusteringFilter(BaseDocumentTransformer, BaseModel):
    """Perform K-means clustering on document vectors.
    Returns an arbitrary number of documents closest to center."""
    embeddings: Embeddings
    'Embeddings to use for embedding document contents.'
    num_clusters: int = 5
    'Number of clusters. Groups of documents with similar meaning.'
    num_closest: int = 1
    'The number of closest vectors to return for each cluster center.'
    random_state: int = 42
    'Controls the random number generator used to initialize the cluster centroids.\n    If you set the random_state parameter to None, the KMeans algorithm will use a \n    random number generator that is seeded with the current time. This means \n    that the results of the KMeans algorithm will be different each time you \n    run it.'
    sorted: bool = False
    'By default results are re-ordered "grouping" them by cluster, if sorted is true\n    result will be ordered by the original position from the retriever'
    remove_duplicates: bool = False
    ' By default duplicated results are skipped and replaced by the next closest \n    vector in the cluster. If remove_duplicates is true no replacement will be done:\n    This could dramatically reduce results when there is a lot of overlap between \n    clusters.\n    '

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Filter down documents."""
        stateful_documents = get_stateful_documents(documents)
        embedded_documents = _get_embeddings_from_stateful_docs(self.embeddings, stateful_documents)
        included_idxs = _filter_cluster_embeddings(embedded_documents, self.num_clusters, self.num_closest, self.random_state, self.remove_duplicates)
        results = sorted(included_idxs) if self.sorted else included_idxs
        return [stateful_documents[i] for i in results]