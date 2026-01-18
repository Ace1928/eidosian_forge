from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
class Clickhouse(VectorStore):
    """`ClickHouse VectorSearch` vector store.

    You need a `clickhouse-connect` python package, and a valid account
    to connect to ClickHouse.

    ClickHouse can not only search with simple vector indexes,
    it also supports complex query with multiple conditions,
    constraints and even sub-queries.

    For more information, please visit
        [ClickHouse official site](https://clickhouse.com/clickhouse)
    """

    def __init__(self, embedding: Embeddings, config: Optional[ClickhouseSettings]=None, **kwargs: Any) -> None:
        """ClickHouse Wrapper to LangChain

        embedding_function (Embeddings):
        config (ClickHouseSettings): Configuration to ClickHouse Client
        Other keyword arguments will pass into
            [clickhouse-connect](https://docs.clickhouse.com/)
        """
        try:
            from clickhouse_connect import get_client
        except ImportError:
            raise ImportError('Could not import clickhouse connect python package. Please install it with `pip install clickhouse-connect`.')
        try:
            from tqdm import tqdm
            self.pgbar = tqdm
        except ImportError:
            self.pgbar = lambda x, **kwargs: x
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = ClickhouseSettings()
        assert self.config
        assert self.config.host and self.config.port
        assert self.config.column_map and self.config.database and self.config.table and self.config.metric
        for k in ['id', 'embedding', 'document', 'metadata', 'uuid']:
            assert k in self.config.column_map
        assert self.config.metric in ['angular', 'euclidean', 'manhattan', 'hamming', 'dot']
        dim = len(embedding.embed_query('test'))
        index_params = (','.join([f"'{k}={v}'" for k, v in self.config.index_param.items()]) if self.config.index_param else '') if isinstance(self.config.index_param, Dict) else ','.join([str(p) for p in self.config.index_param]) if isinstance(self.config.index_param, List) else self.config.index_param
        self.schema = f'CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(\n    {self.config.column_map['id']} Nullable(String),\n    {self.config.column_map['document']} Nullable(String),\n    {self.config.column_map['embedding']} Array(Float32),\n    {self.config.column_map['metadata']} JSON,\n    {self.config.column_map['uuid']} UUID DEFAULT generateUUIDv4(),\n    CONSTRAINT cons_vec_len CHECK length({self.config.column_map['embedding']}) = {dim},\n    INDEX vec_idx {self.config.column_map['embedding']} TYPE {self.config.index_type}({index_params}) GRANULARITY 1000\n) ENGINE = MergeTree ORDER BY uuid SETTINGS index_granularity = 8192'
        self.dim = dim
        self.BS = '\\'
        self.must_escape = ('\\', "'")
        self.embedding_function = embedding
        self.dist_order = 'ASC'
        self.client = get_client(host=self.config.host, port=self.config.port, username=self.config.username, password=self.config.password, **kwargs)
        self.client.command('SET allow_experimental_object_type=1')
        self.client.command(f'SET allow_experimental_{self.config.index_type}_index=1')
        self.client.command(self.schema)

    @property
    def embeddings(self) -> Embeddings:
        """Provides access to the embedding mechanism used by the Clickhouse instance.

        This property allows direct access to the embedding function or model being
        used by the Clickhouse instance to convert text documents into embedding vectors
        for vector similarity search.

        Returns:
            The `Embeddings` instance associated with this Clickhouse instance.
        """
        return self.embedding_function

    def escape_str(self, value: str) -> str:
        """Escape special characters in a string for Clickhouse SQL queries.

        This method is used internally to prepare strings for safe insertion
        into SQL queries by escaping special characters that might otherwise
        interfere with the query syntax.

        Args:
            value: The string to be escaped.

        Returns:
            The escaped string, safe for insertion into SQL queries.
        """
        return ''.join((f'{self.BS}{c}' if c in self.must_escape else c for c in value))

    def _build_insert_sql(self, transac: Iterable, column_names: Iterable[str]) -> str:
        """Construct an SQL query for inserting data into the Clickhouse database.

        This method formats and constructs an SQL `INSERT` query string using the
        provided transaction data and column names. It is utilized internally during
        the process of batch insertion of documents and their embeddings into the
        database.

        Args:
            transac: iterable of tuples, representing a row of data to be inserted.
            column_names: iterable of strings representing the names of the columns
                into which data will be inserted.

        Returns:
            A string containing the constructed SQL `INSERT` query.
        """
        ks = ','.join(column_names)
        _data = []
        for n in transac:
            n = ','.join([f"'{self.escape_str(str(_n))}'" for _n in n])
            _data.append(f'({n})')
        i_str = f'\n                INSERT INTO TABLE \n                    {self.config.database}.{self.config.table}({ks})\n                VALUES\n                {','.join(_data)}\n                '
        return i_str

    def _insert(self, transac: Iterable, column_names: Iterable[str]) -> None:
        """Execute an SQL query to insert data into the Clickhouse database.

        This method performs the actual insertion of data into the database by
        executing the SQL query constructed by `_build_insert_sql`. It's a critical
        step in adding new documents and their associated data into the vector store.

        Args:
            transac:iterable of tuples, representing a row of data to be inserted.
            column_names: An iterable of strings representing the names of the columns
                into which data will be inserted.
        """
        _insert_query = self._build_insert_sql(transac, column_names)
        self.client.command(_insert_query)

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, batch_size: int=32, ids: Optional[Iterable[str]]=None, **kwargs: Any) -> List[str]:
        """Insert more texts through the embeddings and add to the VectorStore.

        Args:
            texts: Iterable of strings to add to the VectorStore.
            ids: Optional list of ids to associate with the texts.
            batch_size: Batch size of insertion
            metadata: Optional column data to be inserted

        Returns:
            List of ids from adding the texts into the VectorStore.

        """
        ids = ids or [sha1(t.encode('utf-8')).hexdigest() for t in texts]
        colmap_ = self.config.column_map
        transac = []
        column_names = {colmap_['id']: ids, colmap_['document']: texts, colmap_['embedding']: self.embedding_function.embed_documents(list(texts))}
        metadatas = metadatas or [{} for _ in texts]
        column_names[colmap_['metadata']] = map(json.dumps, metadatas)
        assert len(set(colmap_) - set(column_names)) >= 0
        keys, values = zip(*column_names.items())
        try:
            t = None
            for v in self.pgbar(zip(*values), desc='Inserting data...', total=len(metadatas)):
                assert len(v[keys.index(self.config.column_map['embedding'])]) == self.dim
                transac.append(v)
                if len(transac) == batch_size:
                    if t:
                        t.join()
                    t = Thread(target=self._insert, args=[transac, keys])
                    t.start()
                    transac = []
            if len(transac) > 0:
                if t:
                    t.join()
                self._insert(transac, keys)
            return [i for i in ids]
        except Exception as e:
            logger.error(f'\x1b[91m\x1b[1m{type(e)}\x1b[0m \x1b[95m{str(e)}\x1b[0m')
            return []

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[Dict[Any, Any]]]=None, config: Optional[ClickhouseSettings]=None, text_ids: Optional[Iterable[str]]=None, batch_size: int=32, **kwargs: Any) -> Clickhouse:
        """Create ClickHouse wrapper with existing texts

        Args:
            embedding_function (Embeddings): Function to extract text embedding
            texts (Iterable[str]): List or tuple of strings to be added
            config (ClickHouseSettings, Optional): ClickHouse configuration
            text_ids (Optional[Iterable], optional): IDs for the texts.
                                                     Defaults to None.
            batch_size (int, optional): Batchsize when transmitting data to ClickHouse.
                                        Defaults to 32.
            metadata (List[dict], optional): metadata to texts. Defaults to None.
            Other keyword arguments will pass into
                [clickhouse-connect](https://clickhouse.com/docs/en/integrations/python#clickhouse-connect-driver-api)
        Returns:
            ClickHouse Index
        """
        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def __repr__(self) -> str:
        """Text representation for ClickHouse Vector Store, prints backends, username
            and schemas. Easy to use with `str(ClickHouse())`

        Returns:
            repr: string to show connection info and data schema
        """
        _repr = f'\x1b[92m\x1b[1m{self.config.database}.{self.config.table} @ '
        _repr += f'{self.config.host}:{self.config.port}\x1b[0m\n\n'
        _repr += f'\x1b[1musername: {self.config.username}\x1b[0m\n\nTable Schema:\n'
        _repr += '-' * 51 + '\n'
        for r in self.client.query(f'DESC {self.config.database}.{self.config.table}').named_results():
            _repr += f'|\x1b[94m{r['name']:24s}\x1b[0m|\x1b[96m{r['type']:24s}\x1b[0m|\n'
        _repr += '-' * 51 + '\n'
        return _repr

    def _build_query_sql(self, q_emb: List[float], topk: int, where_str: Optional[str]=None) -> str:
        """Construct an SQL query for performing a similarity search.

        This internal method generates an SQL query for finding the top-k most similar
        vectors in the database to a given query vector.It allows for optional filtering
        conditions to be applied via a WHERE clause.

        Args:
            q_emb: The query vector as a list of floats.
            topk: The number of top similar items to retrieve.
            where_str: opt str representing additional WHERE conditions for the query
                Defaults to None.

        Returns:
            A string containing the SQL query for the similarity search.
        """
        q_emb_str = ','.join(map(str, q_emb))
        if where_str:
            where_str = f'PREWHERE {where_str}'
        else:
            where_str = ''
        settings_strs = []
        if self.config.index_query_params:
            for k in self.config.index_query_params:
                settings_strs.append(f'SETTING {k}={self.config.index_query_params[k]}')
        q_str = f'\n            SELECT {self.config.column_map['document']}, \n                {self.config.column_map['metadata']}, dist\n            FROM {self.config.database}.{self.config.table}\n            {where_str}\n            ORDER BY L2Distance({self.config.column_map['embedding']}, [{q_emb_str}]) \n                AS dist {self.dist_order}\n            LIMIT {topk} {' '.join(settings_strs)}\n            '
        return q_str

    def similarity_search(self, query: str, k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Document]:
        """Perform a similarity search with ClickHouse

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of Documents
        """
        return self.similarity_search_by_vector(self.embedding_function.embed_query(query), k, where_str, **kwargs)

    def similarity_search_by_vector(self, embedding: List[float], k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Document]:
        """Perform a similarity search with ClickHouse by vectors

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of documents
        """
        q_str = self._build_query_sql(embedding, k, where_str)
        try:
            return [Document(page_content=r[self.config.column_map['document']], metadata=r[self.config.column_map['metadata']]) for r in self.client.query(q_str).named_results()]
        except Exception as e:
            logger.error(f'\x1b[91m\x1b[1m{type(e)}\x1b[0m \x1b[95m{str(e)}\x1b[0m')
            return []

    def similarity_search_with_relevance_scores(self, query: str, k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Perform a similarity search with ClickHouse

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): where condition string.
                                                 Defaults to None.

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection. When dealing with metadatas, remember to
                  use `{self.metadata_column}.attribute` instead of `attribute`
                  alone. The default name for it is `metadata`.

        Returns:
            List[Document]: List of (Document, similarity)
        """
        q_str = self._build_query_sql(self.embedding_function.embed_query(query), k, where_str)
        try:
            return [(Document(page_content=r[self.config.column_map['document']], metadata=r[self.config.column_map['metadata']]), r['dist']) for r in self.client.query(q_str).named_results()]
        except Exception as e:
            logger.error(f'\x1b[91m\x1b[1m{type(e)}\x1b[0m \x1b[95m{str(e)}\x1b[0m')
            return []

    def drop(self) -> None:
        """
        Helper function: Drop data
        """
        self.client.command(f'DROP TABLE IF EXISTS {self.config.database}.{self.config.table}')

    @property
    def metadata_column(self) -> str:
        return self.config.column_map['metadata']