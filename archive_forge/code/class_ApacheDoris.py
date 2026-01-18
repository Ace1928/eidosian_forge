from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
class ApacheDoris(VectorStore):
    """`Apache Doris` vector store.

    You need a `pymysql` python package, and a valid account
    to connect to Apache Doris.

    For more information, please visit
        [Apache Doris official site](https://doris.apache.org/)
        [Apache Doris github](https://github.com/apache/doris)
    """

    def __init__(self, embedding: Embeddings, *, config: Optional[ApacheDorisSettings]=None, **kwargs: Any) -> None:
        """Constructor for Apache Doris.

        Args:
            embedding (Embeddings): Text embedding model.
            config (ApacheDorisSettings): Apache Doris client configuration information.
        """
        try:
            import pymysql
        except ImportError:
            raise ImportError('Could not import pymysql python package. Please install it with `pip install pymysql`.')
        try:
            from tqdm import tqdm
            self.pgbar = tqdm
        except ImportError:
            self.pgbar = lambda x, **kwargs: x
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = ApacheDorisSettings()
        assert self.config
        assert self.config.host and self.config.port
        assert self.config.column_map and self.config.database and self.config.table
        for k in ['id', 'embedding', 'document', 'metadata']:
            assert k in self.config.column_map
        dim = len(embedding.embed_query('test'))
        self.schema = f'CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(    \n    {self.config.column_map['id']} varchar(50),\n    {self.config.column_map['document']} string,\n    {self.config.column_map['embedding']} array<float>,\n    {self.config.column_map['metadata']} string\n) ENGINE = OLAP UNIQUE KEY(id) DISTRIBUTED BY HASH(id)   PROPERTIES ("replication_allocation" = "tag.location.default: 1")'
        self.dim = dim
        self.BS = '\\'
        self.must_escape = ('\\', "'")
        self._embedding = embedding
        self.dist_order = 'DESC'
        _debug_output(self.config)
        self.connection = pymysql.connect(host=self.config.host, port=self.config.port, user=self.config.username, password=self.config.password, database=self.config.database, **kwargs)
        _debug_output(self.schema)
        _get_named_result(self.connection, self.schema)

    def escape_str(self, value: str) -> str:
        return ''.join((f'{self.BS}{c}' if c in self.must_escape else c for c in value))

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _build_insert_sql(self, transac: Iterable, column_names: Iterable[str]) -> str:
        ks = ','.join(column_names)
        embed_tuple_index = tuple(column_names).index(self.config.column_map['embedding'])
        _data = []
        for n in transac:
            n = ','.join([f"'{self.escape_str(str(_n))}'" if idx != embed_tuple_index else f'{str(_n)}' for idx, _n in enumerate(n)])
            _data.append(f'({n})')
        i_str = f'\n                INSERT INTO\n                    {self.config.database}.{self.config.table}({ks})\n                VALUES\n                {','.join(_data)}\n                '
        return i_str

    def _insert(self, transac: Iterable, column_names: Iterable[str]) -> None:
        _insert_query = self._build_insert_sql(transac, column_names)
        _debug_output(_insert_query)
        _get_named_result(self.connection, _insert_query)

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
        column_names = {colmap_['id']: ids, colmap_['document']: texts, colmap_['embedding']: self._embedding.embed_documents(list(texts))}
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
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[Dict[Any, Any]]]=None, config: Optional[ApacheDorisSettings]=None, text_ids: Optional[Iterable[str]]=None, batch_size: int=32, **kwargs: Any) -> ApacheDoris:
        """Create Apache Doris wrapper with existing texts

        Args:
            embedding_function (Embeddings): Function to extract text embedding
            texts (Iterable[str]): List or tuple of strings to be added
            config (ApacheDorisSettings, Optional): Apache Doris configuration
            text_ids (Optional[Iterable], optional): IDs for the texts.
                                                     Defaults to None.
            batch_size (int, optional): BatchSize when transmitting data to Apache
                                        Doris. Defaults to 32.
            metadata (List[dict], optional): metadata to texts. Defaults to None.
        Returns:
            Apache Doris Index
        """
        ctx = cls(embedding, config=config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def __repr__(self) -> str:
        """Text representation for Apache Doris Vector Store, prints frontends, username
            and schemas. Easy to use with `str(ApacheDoris())`

        Returns:
            repr: string to show connection info and data schema
        """
        _repr = f'\x1b[92m\x1b[1m{self.config.database}.{self.config.table} @ '
        _repr += f'{self.config.host}:{self.config.port}\x1b[0m\n\n'
        _repr += f'\x1b[1musername: {self.config.username}\x1b[0m\n\nTable Schema:\n'
        width = 25
        fields = 3
        _repr += '-' * (width * fields + 1) + '\n'
        columns = ['name', 'type', 'key']
        _repr += f'|\x1b[94m{columns[0]:24s}\x1b[0m|\x1b[96m{columns[1]:24s}'
        _repr += f'\x1b[0m|\x1b[96m{columns[2]:24s}\x1b[0m|\n'
        _repr += '-' * (width * fields + 1) + '\n'
        q_str = f'DESC {self.config.database}.{self.config.table}'
        _debug_output(q_str)
        rs = _get_named_result(self.connection, q_str)
        for r in rs:
            _repr += f'|\x1b[94m{r['Field']:24s}\x1b[0m|\x1b[96m{r['Type']:24s}'
            _repr += f'\x1b[0m|\x1b[96m{r['Key']:24s}\x1b[0m|\n'
        _repr += '-' * (width * fields + 1) + '\n'
        return _repr

    def _build_query_sql(self, q_emb: List[float], topk: int, where_str: Optional[str]=None) -> str:
        q_emb_str = ','.join(map(str, q_emb))
        if where_str:
            where_str = f'WHERE {where_str}'
        else:
            where_str = ''
        q_str = f'\n            SELECT {self.config.column_map['document']}, \n                {self.config.column_map['metadata']}, \n                cosine_distance(array<float>[{q_emb_str}],\n                  {self.config.column_map['embedding']}) as dist\n            FROM {self.config.database}.{self.config.table}\n            {where_str}\n            ORDER BY dist {self.dist_order}\n            LIMIT {topk}\n            '
        _debug_output(q_str)
        return q_str

    def similarity_search(self, query: str, k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Document]:
        """Perform a similarity search with Apache Doris

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
        return self.similarity_search_by_vector(self._embedding.embed_query(query), k, where_str, **kwargs)

    def similarity_search_by_vector(self, embedding: List[float], k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Document]:
        """Perform a similarity search with Apache Doris by vectors

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
        q_str = self._build_query_sql(embedding, k, where_str)
        try:
            return [Document(page_content=r[self.config.column_map['document']], metadata=json.loads(r[self.config.column_map['metadata']])) for r in _get_named_result(self.connection, q_str)]
        except Exception as e:
            logger.error(f'\x1b[91m\x1b[1m{type(e)}\x1b[0m \x1b[95m{str(e)}\x1b[0m')
            return []

    def similarity_search_with_relevance_scores(self, query: str, k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Apache Doris

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
        q_str = self._build_query_sql(self._embedding.embed_query(query), k, where_str)
        try:
            return [(Document(page_content=r[self.config.column_map['document']], metadata=json.loads(r[self.config.column_map['metadata']])), r['dist']) for r in _get_named_result(self.connection, q_str)]
        except Exception as e:
            logger.error(f'\x1b[91m\x1b[1m{type(e)}\x1b[0m \x1b[95m{str(e)}\x1b[0m')
            return []

    def drop(self) -> None:
        """
        Helper function: Drop data
        """
        _get_named_result(self.connection, f'DROP TABLE IF EXISTS {self.config.database}.{self.config.table}')

    @property
    def metadata_column(self) -> str:
        return self.config.column_map['metadata']