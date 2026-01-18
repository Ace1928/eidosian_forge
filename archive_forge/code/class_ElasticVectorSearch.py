from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
@deprecated('0.0.27', alternative='Use ElasticsearchStore class in langchain-elasticsearch package', pending=True)
class ElasticVectorSearch(VectorStore):
    """

    ElasticVectorSearch uses the brute force method of searching on vectors.

    Recommended to use ElasticsearchStore instead, which gives you the option
    to uses the approx  HNSW algorithm which performs better on large datasets.

    ElasticsearchStore also supports metadata filtering, customising the
    query retriever and much more!

    You can read more on ElasticsearchStore:
    https://python.langchain.com/docs/integrations/vectorstores/elasticsearch

    To connect to an `Elasticsearch` instance that does not require
    login credentials, pass the Elasticsearch URL and index name along with the
    embedding object to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import ElasticVectorSearch
            from langchain_community.embeddings import OpenAIEmbeddings

            embedding = OpenAIEmbeddings()
            elastic_vector_search = ElasticVectorSearch(
                elasticsearch_url="http://localhost:9200",
                index_name="test_index",
                embedding=embedding
            )


    To connect to an Elasticsearch instance that requires login credentials,
    including Elastic Cloud, use the Elasticsearch URL format
    https://username:password@es_host:9243. For example, to connect to Elastic
    Cloud, create the Elasticsearch URL with the required authentication details and
    pass it to the ElasticVectorSearch constructor as the named parameter
    elasticsearch_url.

    You can obtain your Elastic Cloud URL and login credentials by logging in to the
    Elastic Cloud console at https://cloud.elastic.co, selecting your deployment, and
    navigating to the "Deployments" page.

    To obtain your Elastic Cloud password for the default "elastic" user:

    1. Log in to the Elastic Cloud console at https://cloud.elastic.co
    2. Go to "Security" > "Users"
    3. Locate the "elastic" user and click "Edit"
    4. Click "Reset password"
    5. Follow the prompts to reset the password

    The format for Elastic Cloud URLs is
    https://username:password@cluster_id.region_id.gcp.cloud.es.io:9243.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import ElasticVectorSearch
            from langchain_community.embeddings import OpenAIEmbeddings

            embedding = OpenAIEmbeddings()

            elastic_host = "cluster_id.region_id.gcp.cloud.es.io"
            elasticsearch_url = f"https://username:password@{elastic_host}:9243"
            elastic_vector_search = ElasticVectorSearch(
                elasticsearch_url=elasticsearch_url,
                index_name="test_index",
                embedding=embedding
            )

    Args:
        elasticsearch_url (str): The URL for the Elasticsearch instance.
        index_name (str): The name of the Elasticsearch index for the embeddings.
        embedding (Embeddings): An object that provides the ability to embed text.
                It should be an instance of a class that subclasses the Embeddings
                abstract base class, such as OpenAIEmbeddings()

    Raises:
        ValueError: If the elasticsearch python package is not installed.
    """

    def __init__(self, elasticsearch_url: str, index_name: str, embedding: Embeddings, *, ssl_verify: Optional[Dict[str, Any]]=None):
        """Initialize with necessary components."""
        warnings.warn('ElasticVectorSearch will be removed in a future release. SeeElasticsearch integration docs on how to upgrade.')
        try:
            import elasticsearch
        except ImportError:
            raise ImportError('Could not import elasticsearch python package. Please install it with `pip install elasticsearch`.')
        self.embedding = embedding
        self.index_name = index_name
        _ssl_verify = ssl_verify or {}
        try:
            self.client = elasticsearch.Elasticsearch(elasticsearch_url, **_ssl_verify, headers={'user-agent': self.get_user_agent()})
        except ValueError as e:
            raise ValueError(f'Your elasticsearch client string is mis-formatted. Got error: {e} ')

    @staticmethod
    def get_user_agent() -> str:
        from langchain_community import __version__
        return f'langchain-py-dvs/{__version__}'

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, refresh_indices: bool=True, **kwargs: Any) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        try:
            from elasticsearch.exceptions import NotFoundError
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ImportError('Could not import elasticsearch python package. Please install it with `pip install elasticsearch`.')
        requests = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embedding.embed_documents(list(texts))
        dim = len(embeddings[0])
        mapping = _default_text_mapping(dim)
        try:
            self.client.indices.get(index=self.index_name)
        except NotFoundError:
            self.create_index(self.client, self.index_name, mapping)
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            request = {'_op_type': 'index', '_index': self.index_name, 'vector': embeddings[i], 'text': text, 'metadata': metadata, '_id': ids[i]}
            requests.append(request)
        bulk(self.client, requests)
        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids

    def similarity_search(self, query: str, k: int=4, filter: Optional[dict]=None, **kwargs: Any) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        documents = [d[0] for d in docs_and_scores]
        return documents

    def similarity_search_with_score(self, query: str, k: int=4, filter: Optional[dict]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding.embed_query(query)
        script_query = _default_script_query(embedding, filter)
        response = self.client_search(self.client, self.index_name, script_query, size=k)
        hits = [hit for hit in response['hits']['hits']]
        docs_and_scores = [(Document(page_content=hit['_source']['text'], metadata=hit['_source']['metadata']), hit['_score']) for hit in hits]
        return docs_and_scores

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, index_name: Optional[str]=None, refresh_indices: bool=True, **kwargs: Any) -> ElasticVectorSearch:
        """Construct ElasticVectorSearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the Elasticsearch instance.
            3. Adds the documents to the newly created Elasticsearch index.

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import ElasticVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                elastic_vector_search = ElasticVectorSearch.from_texts(
                    texts,
                    embeddings,
                    elasticsearch_url="http://localhost:9200"
                )
        """
        elasticsearch_url = get_from_dict_or_env(kwargs, 'elasticsearch_url', 'ELASTICSEARCH_URL')
        if 'elasticsearch_url' in kwargs:
            del kwargs['elasticsearch_url']
        index_name = index_name or uuid.uuid4().hex
        vectorsearch = cls(elasticsearch_url, index_name, embedding, **kwargs)
        vectorsearch.add_texts(texts, metadatas=metadatas, ids=ids, refresh_indices=refresh_indices)
        return vectorsearch

    def create_index(self, client: Any, index_name: str, mapping: Dict) -> None:
        version_num = client.info()['version']['number'][0]
        version_num = int(version_num)
        if version_num >= 8:
            client.indices.create(index=index_name, mappings=mapping)
        else:
            client.indices.create(index=index_name, body={'mappings': mapping})

    def client_search(self, client: Any, index_name: str, script_query: Dict, size: int) -> Any:
        version_num = client.info()['version']['number'][0]
        version_num = int(version_num)
        if version_num >= 8:
            response = client.search(index=index_name, query=script_query, size=size)
        else:
            response = client.search(index=index_name, body={'query': script_query, 'size': size})
        return response

    def delete(self, ids: Optional[List[str]]=None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        if ids is None:
            raise ValueError('No ids provided to delete.')
        for id in ids:
            self.client.delete(index=self.index_name, id=id)