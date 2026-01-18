from typing import Any, Callable, List, Sequence
import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utils.math import cosine_similarity
def _filter_cluster_embeddings(embedded_documents: List[List[float]], num_clusters: int, num_closest: int, random_state: int, remove_duplicates: bool) -> List[int]:
    """Filter documents based on proximity of their embeddings to clusters."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError('sklearn package not found, please install it with `pip install scikit-learn`')
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(embedded_documents)
    closest_indices = []
    for i in range(num_clusters):
        distances = np.linalg.norm(embedded_documents - kmeans.cluster_centers_[i], axis=1)
        if remove_duplicates:
            closest_indices_sorted = [x for x in np.argsort(distances)[:num_closest] if x not in closest_indices]
        else:
            closest_indices_sorted = [x for x in np.argsort(distances) if x not in closest_indices][:num_closest]
        closest_indices.extend(closest_indices_sorted)
    return closest_indices