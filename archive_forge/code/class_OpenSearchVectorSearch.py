from __future__ import annotations
import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class OpenSearchVectorSearch(VectorStore):
    """`Amazon OpenSearch Vector Engine` vector store.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import OpenSearchVectorSearch
            opensearch_vector_search = OpenSearchVectorSearch(
                "http://localhost:9200",
                "embeddings",
                embedding_function
            )

    """

    def __init__(self, opensearch_url: str, index_name: str, embedding_function: Embeddings, **kwargs: Any):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index_name = index_name
        http_auth = kwargs.get('http_auth')
        self.is_aoss = _is_aoss_enabled(http_auth=http_auth)
        self.client = _get_opensearch_client(opensearch_url, **kwargs)
        self.async_client = _get_async_opensearch_client(opensearch_url, **kwargs)
        self.engine = kwargs.get('engine')

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def __add(self, texts: Iterable[str], embeddings: List[List[float]], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, bulk_size: int=500, **kwargs: Any) -> List[str]:
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        index_name = kwargs.get('index_name', self.index_name)
        text_field = kwargs.get('text_field', 'text')
        dim = len(embeddings[0])
        engine = kwargs.get('engine', 'nmslib')
        space_type = kwargs.get('space_type', 'l2')
        ef_search = kwargs.get('ef_search', 512)
        ef_construction = kwargs.get('ef_construction', 512)
        m = kwargs.get('m', 16)
        vector_field = kwargs.get('vector_field', 'vector_field')
        max_chunk_bytes = kwargs.get('max_chunk_bytes', 1 * 1024 * 1024)
        _validate_aoss_with_engines(self.is_aoss, engine)
        mapping = _default_text_mapping(dim, engine, space_type, ef_search, ef_construction, m, vector_field)
        return _bulk_ingest_embeddings(self.client, index_name, embeddings, texts, metadatas=metadatas, ids=ids, vector_field=vector_field, text_field=text_field, mapping=mapping, max_chunk_bytes=max_chunk_bytes, is_aoss=self.is_aoss)

    async def __aadd(self, texts: Iterable[str], embeddings: List[List[float]], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, bulk_size: int=500, **kwargs: Any) -> List[str]:
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        index_name = kwargs.get('index_name', self.index_name)
        text_field = kwargs.get('text_field', 'text')
        dim = len(embeddings[0])
        engine = kwargs.get('engine', 'nmslib')
        space_type = kwargs.get('space_type', 'l2')
        ef_search = kwargs.get('ef_search', 512)
        ef_construction = kwargs.get('ef_construction', 512)
        m = kwargs.get('m', 16)
        vector_field = kwargs.get('vector_field', 'vector_field')
        max_chunk_bytes = kwargs.get('max_chunk_bytes', 1 * 1024 * 1024)
        _validate_aoss_with_engines(self.is_aoss, engine)
        mapping = _default_text_mapping(dim, engine, space_type, ef_search, ef_construction, m, vector_field)
        return await _abulk_ingest_embeddings(self.async_client, index_name, embeddings, texts, metadatas=metadatas, ids=ids, vector_field=vector_field, text_field=text_field, mapping=mapping, max_chunk_bytes=max_chunk_bytes, is_aoss=self.is_aoss)

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, bulk_size: int=500, **kwargs: Any) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            bulk_size: Bulk API request count; Default: 500

        Returns:
            List of ids from adding the texts into the vectorstore.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".
        """
        embeddings = self.embedding_function.embed_documents(list(texts))
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, bulk_size=bulk_size, **kwargs)

    async def aadd_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, bulk_size: int=500, **kwargs: Any) -> List[str]:
        """
        Asynchronously run more texts through the embeddings
        and add to the vectorstore.
        """
        embeddings = await self.embedding_function.aembed_documents(list(texts))
        return await self.__aadd(texts, embeddings, metadatas=metadatas, ids=ids, bulk_size=bulk_size, **kwargs)

    def add_embeddings(self, text_embeddings: Iterable[Tuple[str, List[float]]], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, bulk_size: int=500, **kwargs: Any) -> List[str]:
        """Add the given texts and embeddings to the vectorstore.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            bulk_size: Bulk API request count; Default: 500

        Returns:
            List of ids from adding the texts into the vectorstore.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".
        """
        texts, embeddings = zip(*text_embeddings)
        return self.__add(list(texts), list(embeddings), metadatas=metadatas, ids=ids, bulk_size=bulk_size, **kwargs)

    def delete(self, ids: Optional[List[str]]=None, refresh_indices: Optional[bool]=True, **kwargs: Any) -> Optional[bool]:
        """Delete documents from the Opensearch index.

        Args:
            ids: List of ids of documents to delete.
            refresh_indices: Whether to refresh the index
                            after deleting documents. Defaults to True.
        """
        bulk = _import_bulk()
        body = []
        if ids is None:
            raise ValueError('ids must be provided.')
        for _id in ids:
            body.append({'_op_type': 'delete', '_index': self.index_name, '_id': _id})
        if len(body) > 0:
            try:
                bulk(self.client, body, refresh=refresh_indices, ignore_status=404)
                return True
            except Exception as e:
                raise e
        else:
            return False

    async def adelete(self, ids: Optional[List[str]]=None, **kwargs: Any) -> Optional[bool]:
        """Asynchronously delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            raise ValueError('No ids provided to delete.')
        actions = [{'delete': {'_index': self.index_name, '_id': id_}} for id_ in ids]
        response = await self.async_client.bulk(body=actions, **kwargs)
        return not any((item.get('delete', {}).get('error') for item in response['items']))

    def similarity_search(self, query: str, k: int=4, **kwargs: Any) -> List[Document]:
        """Return docs most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

            metadata_field: Document field that metadata is stored in. Defaults to
            "metadata".
            Can be set to a special value "*" to include the entire document.

        Optional Args for Approximate Search:
            search_type: "approximate_search"; default: "approximate_search"

            boolean_filter: A Boolean filter is a post filter consists of a Boolean
            query that contains a k-NN query and a filter.

            subquery_clause: Query clause on the knn vector field; default: "must"

            lucene_filter: the Lucene algorithm decides whether to perform an exact
            k-NN search with pre-filtering or an approximate search with modified
            post-filtering. (deprecated, use `efficient_filter`)

            efficient_filter: the Lucene Engine or Faiss Engine decides whether to
            perform an exact k-NN search with pre-filtering or an approximate search
            with modified post-filtering.

        Optional Args for Script Scoring Search:
            search_type: "script_scoring"; default: "approximate_search"

            space_type: "l2", "l1", "linf", "cosinesimil", "innerproduct",
            "hammingbit"; default: "l2"

            pre_filter: script_score query to pre-filter documents before identifying
            nearest neighbors; default: {"match_all": {}}

        Optional Args for Painless Scripting Search:
            search_type: "painless_scripting"; default: "approximate_search"

            space_type: "l2Squared", "l1Norm", "cosineSimilarity"; default: "l2Squared"

            pre_filter: script_score query to pre-filter documents before identifying
            nearest neighbors; default: {"match_all": {}}
        """
        docs_with_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc[0] for doc in docs_with_scores]

    def similarity_search_by_vector(self, embedding: List[float], k: int=4, **kwargs: Any) -> List[Document]:
        """Return docs most similar to the embedding vector."""
        docs_with_scores = self.similarity_search_with_score_by_vector(embedding, k, **kwargs)
        return [doc[0] for doc in docs_with_scores]

    def similarity_search_with_score(self, query: str, k: int=4, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Return docs and it's scores most similar to query.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents along with its scores most similar to the query.

        Optional Args:
            same as `similarity_search`
        """
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_with_score_by_vector(embedding, k, **kwargs)

    def similarity_search_with_score_by_vector(self, embedding: List[float], k: int=4, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Return docs and it's scores most similar to the embedding vector.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents along with its scores most similar to the query.

        Optional Args:
            same as `similarity_search`
        """
        text_field = kwargs.get('text_field', 'text')
        metadata_field = kwargs.get('metadata_field', 'metadata')
        hits = self._raw_similarity_search_with_score_by_vector(embedding=embedding, k=k, **kwargs)
        documents_with_scores = [(Document(page_content=hit['_source'][text_field], metadata=hit['_source'] if metadata_field == '*' or metadata_field not in hit['_source'] else hit['_source'][metadata_field]), hit['_score']) for hit in hits]
        return documents_with_scores

    def _raw_similarity_search_with_score_by_vector(self, embedding: List[float], k: int=4, **kwargs: Any) -> List[dict]:
        """Return raw opensearch documents (dict) including vectors,
        scores most similar to the embedding vector.

        By default, supports Approximate Search.
        Also supports Script Scoring and Painless Scripting.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of dict with its scores most similar to the embedding.

        Optional Args:
            same as `similarity_search`
        """
        search_type = kwargs.get('search_type', 'approximate_search')
        vector_field = kwargs.get('vector_field', 'vector_field')
        index_name = kwargs.get('index_name', self.index_name)
        filter = kwargs.get('filter', {})
        if self.is_aoss and search_type != 'approximate_search' and (search_type != SCRIPT_SCORING_SEARCH):
            raise ValueError('Amazon OpenSearch Service Serverless only supports `approximate_search` and `script_scoring`')
        if search_type == 'approximate_search':
            boolean_filter = kwargs.get('boolean_filter', {})
            subquery_clause = kwargs.get('subquery_clause', 'must')
            efficient_filter = kwargs.get('efficient_filter', {})
            lucene_filter = kwargs.get('lucene_filter', {})
            if boolean_filter != {} and efficient_filter != {}:
                raise ValueError('Both `boolean_filter` and `efficient_filter` are provided which is invalid')
            if lucene_filter != {} and efficient_filter != {}:
                raise ValueError('Both `lucene_filter` and `efficient_filter` are provided which is invalid. `lucene_filter` is deprecated')
            if lucene_filter != {} and boolean_filter != {}:
                raise ValueError('Both `lucene_filter` and `boolean_filter` are provided which is invalid. `lucene_filter` is deprecated')
            if efficient_filter == {} and boolean_filter == {} and (lucene_filter == {}) and (filter != {}):
                if self.engine in ['faiss', 'lucene']:
                    efficient_filter = filter
                else:
                    boolean_filter = filter
            if boolean_filter != {}:
                search_query = _approximate_search_query_with_boolean_filter(embedding, boolean_filter, k=k, vector_field=vector_field, subquery_clause=subquery_clause)
            elif efficient_filter != {}:
                search_query = _approximate_search_query_with_efficient_filter(embedding, efficient_filter, k=k, vector_field=vector_field)
            elif lucene_filter != {}:
                warnings.warn('`lucene_filter` is deprecated. Please use the keyword argument `efficient_filter`')
                search_query = _approximate_search_query_with_efficient_filter(embedding, lucene_filter, k=k, vector_field=vector_field)
            else:
                search_query = _default_approximate_search_query(embedding, k=k, vector_field=vector_field)
        elif search_type == SCRIPT_SCORING_SEARCH:
            space_type = kwargs.get('space_type', 'l2')
            pre_filter = kwargs.get('pre_filter', MATCH_ALL_QUERY)
            search_query = _default_script_query(embedding, k, space_type, pre_filter, vector_field)
        elif search_type == PAINLESS_SCRIPTING_SEARCH:
            space_type = kwargs.get('space_type', 'l2Squared')
            pre_filter = kwargs.get('pre_filter', MATCH_ALL_QUERY)
            search_query = _default_painless_scripting_query(embedding, k, space_type, pre_filter, vector_field)
        else:
            raise ValueError('Invalid `search_type` provided as an argument')
        response = self.client.search(index=index_name, body=search_query)
        return [hit for hit in response['hits']['hits']]

    def max_marginal_relevance_search(self, query: str, k: int=4, fetch_k: int=20, lambda_mult: float=0.5, **kwargs: Any) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        vector_field = kwargs.get('vector_field', 'vector_field')
        text_field = kwargs.get('text_field', 'text')
        metadata_field = kwargs.get('metadata_field', 'metadata')
        embedding = self.embedding_function.embed_query(query)
        results = self._raw_similarity_search_with_score_by_vector(embedding, fetch_k, **kwargs)
        embeddings = [result['_source'][vector_field] for result in results]
        mmr_selected = maximal_marginal_relevance(np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult)
        return [Document(page_content=results[i]['_source'][text_field], metadata=results[i]['_source'][metadata_field]) for i in mmr_selected]

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, bulk_size: int=500, ids: Optional[List[str]]=None, **kwargs: Any) -> OpenSearchVectorSearch:
        """Construct OpenSearchVectorSearch wrapper from raw texts.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                opensearch_vector_search = OpenSearchVectorSearch.from_texts(
                    texts,
                    embeddings,
                    opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        embeddings = embedding.embed_documents(texts)
        return cls.from_embeddings(embeddings, texts, embedding, metadatas=metadatas, bulk_size=bulk_size, ids=ids, **kwargs)

    @classmethod
    async def afrom_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, bulk_size: int=500, ids: Optional[List[str]]=None, **kwargs: Any) -> OpenSearchVectorSearch:
        """Asynchronously construct OpenSearchVectorSearch wrapper from raw texts.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                opensearch_vector_search = await OpenSearchVectorSearch.afrom_texts(
                    texts,
                    embeddings,
                    opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        embeddings = await embedding.aembed_documents(texts)
        return await cls.afrom_embeddings(embeddings, texts, embedding, metadatas=metadatas, bulk_size=bulk_size, ids=ids, **kwargs)

    @classmethod
    def from_embeddings(cls, embeddings: List[List[float]], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, bulk_size: int=500, ids: Optional[List[str]]=None, **kwargs: Any) -> OpenSearchVectorSearch:
        """Construct OpenSearchVectorSearch wrapper from pre-vectorized embeddings.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embedder = OpenAIEmbeddings()
                embeddings = embedder.embed_documents(["foo", "bar"])
                opensearch_vector_search = OpenSearchVectorSearch.from_embeddings(
                    embeddings,
                    texts,
                    embedder,
                    opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        opensearch_url = get_from_dict_or_env(kwargs, 'opensearch_url', 'OPENSEARCH_URL')
        keys_list = ['opensearch_url', 'index_name', 'is_appx_search', 'vector_field', 'text_field', 'engine', 'space_type', 'ef_search', 'ef_construction', 'm', 'max_chunk_bytes', 'is_aoss']
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        dim = len(embeddings[0])
        index_name = get_from_dict_or_env(kwargs, 'index_name', 'OPENSEARCH_INDEX_NAME', default=uuid.uuid4().hex)
        is_appx_search = kwargs.get('is_appx_search', True)
        vector_field = kwargs.get('vector_field', 'vector_field')
        text_field = kwargs.get('text_field', 'text')
        max_chunk_bytes = kwargs.get('max_chunk_bytes', 1 * 1024 * 1024)
        http_auth = kwargs.get('http_auth')
        is_aoss = _is_aoss_enabled(http_auth=http_auth)
        engine = None
        if is_aoss and (not is_appx_search):
            raise ValueError('Amazon OpenSearch Service Serverless only supports `approximate_search`')
        if is_appx_search:
            engine = kwargs.get('engine', 'nmslib')
            space_type = kwargs.get('space_type', 'l2')
            ef_search = kwargs.get('ef_search', 512)
            ef_construction = kwargs.get('ef_construction', 512)
            m = kwargs.get('m', 16)
            _validate_aoss_with_engines(is_aoss, engine)
            mapping = _default_text_mapping(dim, engine, space_type, ef_search, ef_construction, m, vector_field)
        else:
            mapping = _default_scripting_text_mapping(dim)
        [kwargs.pop(key, None) for key in keys_list]
        client = _get_opensearch_client(opensearch_url, **kwargs)
        _bulk_ingest_embeddings(client, index_name, embeddings, texts, ids=ids, metadatas=metadatas, vector_field=vector_field, text_field=text_field, mapping=mapping, max_chunk_bytes=max_chunk_bytes, is_aoss=is_aoss)
        kwargs['engine'] = engine
        return cls(opensearch_url, index_name, embedding, **kwargs)

    @classmethod
    async def afrom_embeddings(cls, embeddings: List[List[float]], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, bulk_size: int=500, ids: Optional[List[str]]=None, **kwargs: Any) -> OpenSearchVectorSearch:
        """Asynchronously construct OpenSearchVectorSearch wrapper from pre-vectorized
        embeddings.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embedder = OpenAIEmbeddings()
                embeddings = await embedder.aembed_documents(["foo", "bar"])
                opensearch_vector_search =
                    await OpenSearchVectorSearch.afrom_embeddings(
                        embeddings,
                        texts,
                        embedder,
                        opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        opensearch_url = get_from_dict_or_env(kwargs, 'opensearch_url', 'OPENSEARCH_URL')
        keys_list = ['opensearch_url', 'index_name', 'is_appx_search', 'vector_field', 'text_field', 'engine', 'space_type', 'ef_search', 'ef_construction', 'm', 'max_chunk_bytes', 'is_aoss']
        _validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        dim = len(embeddings[0])
        index_name = get_from_dict_or_env(kwargs, 'index_name', 'OPENSEARCH_INDEX_NAME', default=uuid.uuid4().hex)
        is_appx_search = kwargs.get('is_appx_search', True)
        vector_field = kwargs.get('vector_field', 'vector_field')
        text_field = kwargs.get('text_field', 'text')
        max_chunk_bytes = kwargs.get('max_chunk_bytes', 1 * 1024 * 1024)
        http_auth = kwargs.get('http_auth')
        is_aoss = _is_aoss_enabled(http_auth=http_auth)
        engine = None
        if is_aoss and (not is_appx_search):
            raise ValueError('Amazon OpenSearch Service Serverless only supports `approximate_search`')
        if is_appx_search:
            engine = kwargs.get('engine', 'nmslib')
            space_type = kwargs.get('space_type', 'l2')
            ef_search = kwargs.get('ef_search', 512)
            ef_construction = kwargs.get('ef_construction', 512)
            m = kwargs.get('m', 16)
            _validate_aoss_with_engines(is_aoss, engine)
            mapping = _default_text_mapping(dim, engine, space_type, ef_search, ef_construction, m, vector_field)
        else:
            mapping = _default_scripting_text_mapping(dim)
        [kwargs.pop(key, None) for key in keys_list]
        client = _get_async_opensearch_client(opensearch_url, **kwargs)
        await _abulk_ingest_embeddings(client, index_name, embeddings, texts, ids=ids, metadatas=metadatas, vector_field=vector_field, text_field=text_field, mapping=mapping, max_chunk_bytes=max_chunk_bytes, is_aoss=is_aoss)
        kwargs['engine'] = engine
        return cls(opensearch_url, index_name, embedding, **kwargs)