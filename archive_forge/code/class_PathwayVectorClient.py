import json
import logging
from typing import Any, Callable, Iterable, List, Optional, Tuple
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
class PathwayVectorClient(VectorStore):
    """
    VectorStore connecting to Pathway Vector Store.
    """

    def __init__(self, host: Optional[str]=None, port: Optional[int]=None, url: Optional[str]=None) -> None:
        """
        A client you can use to query Pathway Vector Store.

        Please provide aither the `url`, or `host` and `port`.

        Args:
            - host: host on which Pathway Vector Store listens
            - port: port on which Pathway Vector Store listens
            - url: url at which Pathway Vector Store listens
        """
        self.client = _VectorStoreClient(host, port, url)

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, **kwargs: Any) -> List[str]:
        """Pathway is not suitable for this method."""
        raise NotImplementedError('Pathway vector store does not support adding or removing texts from client.')

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, **kwargs: Any) -> 'PathwayVectorClient':
        raise NotImplementedError('Pathway vector store does not support initializing from_texts.')

    def similarity_search(self, query: str, k: int=4, **kwargs: Any) -> List[Document]:
        metadata_filter = kwargs.pop('metadata_filter', None)
        if kwargs:
            logging.warning('Unknown kwargs passed to PathwayVectorClient.similarity_search: %s', kwargs)
        rets = self.client(query=query, k=k, metadata_filter=metadata_filter)
        return [Document(page_content=ret['text'], metadata=ret['metadata']) for ret in rets]

    def similarity_search_with_score(self, query: str, k: int=4, metadata_filter: Optional[str]=None) -> List[Tuple[Document, float]]:
        """Run similarity search with Pathway with distance.

        Args:
            - query (str): Query text to search for.
            - k (int): Number of results to return. Defaults to 4.
            - metadata_filter (Optional[str]): Filter by metadata.
                Filtering query should be in JMESPath format. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        rets = self.client(query=query, k=k, metadata_filter=metadata_filter)
        return [(Document(page_content=ret['text'], metadata=ret['metadata']), ret['dist']) for ret in rets]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._cosine_relevance_score_fn

    def get_vectorstore_statistics(self) -> dict:
        """Fetch basic statistics about the Vector Store."""
        return self.client.get_vectorstore_statistics()

    def get_input_files(self, metadata_filter: Optional[str]=None, filepath_globpattern: Optional[str]=None) -> list:
        """List files indexed by the Vector Store."""
        return self.client.get_input_files(metadata_filter, filepath_globpattern)