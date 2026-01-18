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
class MyScale(VectorStore):
    """`MyScale` vector store.

    You need a `clickhouse-connect` python package, and a valid account
    to connect to MyScale.

    MyScale can not only search with simple vector indexes.
    It also supports a complex query with multiple conditions,
    constraints and even sub-queries.

    For more information, please visit
        [myscale official site](https://docs.myscale.com/en/overview/)
    """

    def __init__(self, embedding: Embeddings, config: Optional[MyScaleSettings]=None, **kwargs: Any) -> None:
        """MyScale Wrapper to LangChain

        embedding (Embeddings):
        config (MyScaleSettings): Configuration to MyScale Client
        Other keyword arguments will pass into
            [clickhouse-connect](https://docs.myscale.com/)
        """
        try:
            from clickhouse_connect import get_client
        except ImportError:
            raise ImportError('Could not import clickhouse connect python package. Please install it with `pip install clickhouse-connect`.')
        try:
            from tqdm import tqdm
            self.pgbar = tqdm
        except ImportError:
            self.pgbar = lambda x: x
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MyScaleSettings()
        assert self.config
        assert self.config.host and self.config.port
        assert self.config.column_map and self.config.database and self.config.table and self.config.metric
        for k in ['id', 'vector', 'text', 'metadata']:
            assert k in self.config.column_map
        assert self.config.metric.upper() in ['IP', 'COSINE', 'L2']
        if self.config.metric in ['ip', 'cosine', 'l2']:
            logger.warning("Lower case metric types will be deprecated the future. Please use one of ('IP', 'Cosine', 'L2')")
        dim = len(embedding.embed_query('try this out'))
        index_params = ', ' + ','.join([f"'{k}={v}'" for k, v in self.config.index_param.items()]) if self.config.index_param else ''
        schema_ = f"\n            CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.table}(\n                {self.config.column_map['id']} String,\n                {self.config.column_map['text']} String,\n                {self.config.column_map['vector']} Array(Float32),\n                {self.config.column_map['metadata']} JSON,\n                CONSTRAINT cons_vec_len CHECK length(                    {self.config.column_map['vector']}) = {dim},\n                VECTOR INDEX vidx {self.config.column_map['vector']}                     TYPE {self.config.index_type}(                        'metric_type={self.config.metric}'{index_params})\n            ) ENGINE = MergeTree ORDER BY {self.config.column_map['id']}\n        "
        self.dim = dim
        self.BS = '\\'
        self.must_escape = ('\\', "'")
        self._embeddings = embedding
        self.dist_order = 'ASC' if self.config.metric.upper() in ['COSINE', 'L2'] else 'DESC'
        self.client = get_client(host=self.config.host, port=self.config.port, username=self.config.username, password=self.config.password, **kwargs)
        self.client.command('SET allow_experimental_object_type=1')
        self.client.command(schema_)

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def escape_str(self, value: str) -> str:
        return ''.join((f'{self.BS}{c}' if c in self.must_escape else c for c in value))

    def _build_istr(self, transac: Iterable, column_names: Iterable[str]) -> str:
        ks = ','.join(column_names)
        _data = []
        for n in transac:
            n = ','.join([f"'{self.escape_str(str(_n))}'" for _n in n])
            _data.append(f'({n})')
        i_str = f'\n                INSERT INTO TABLE \n                    {self.config.database}.{self.config.table}({ks})\n                VALUES\n                {','.join(_data)}\n                '
        return i_str

    def _insert(self, transac: Iterable, column_names: Iterable[str]) -> None:
        _i_str = self._build_istr(transac, column_names)
        self.client.command(_i_str)

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]]=None, batch_size: int=32, ids: Optional[Iterable[str]]=None, **kwargs: Any) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            ids: Optional list of ids to associate with the texts.
            batch_size: Batch size of insertion
            metadata: Optional column data to be inserted

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        ids = ids or [sha1(t.encode('utf-8')).hexdigest() for t in texts]
        colmap_ = self.config.column_map
        transac = []
        column_names = {colmap_['id']: ids, colmap_['text']: texts, colmap_['vector']: map(self._embeddings.embed_query, texts)}
        metadatas = metadatas or [{} for _ in texts]
        column_names[colmap_['metadata']] = map(json.dumps, metadatas)
        assert len(set(colmap_) - set(column_names)) >= 0
        keys, values = zip(*column_names.items())
        try:
            t = None
            for v in self.pgbar(zip(*values), desc='Inserting data...', total=len(metadatas)):
                assert len(v[keys.index(self.config.column_map['vector'])]) == self.dim
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
    def from_texts(cls, texts: Iterable[str], embedding: Embeddings, metadatas: Optional[List[Dict[Any, Any]]]=None, config: Optional[MyScaleSettings]=None, text_ids: Optional[Iterable[str]]=None, batch_size: int=32, **kwargs: Any) -> MyScale:
        """Create Myscale wrapper with existing texts

        Args:
            texts (Iterable[str]): List or tuple of strings to be added
            embedding (Embeddings): Function to extract text embedding
            config (MyScaleSettings, Optional): Myscale configuration
            text_ids (Optional[Iterable], optional): IDs for the texts.
                                                     Defaults to None.
            batch_size (int, optional): Batchsize when transmitting data to MyScale.
                                        Defaults to 32.
            metadata (List[dict], optional): metadata to texts. Defaults to None.
            Other keyword arguments will pass into
                [clickhouse-connect](https://clickhouse.com/docs/en/integrations/python#clickhouse-connect-driver-api)
        Returns:
            MyScale Index
        """
        ctx = cls(embedding, config, **kwargs)
        ctx.add_texts(texts, ids=text_ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def __repr__(self) -> str:
        """Text representation for myscale, prints backends, username and schemas.
            Easy to use with `str(Myscale())`

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

    def _build_qstr(self, q_emb: List[float], topk: int, where_str: Optional[str]=None) -> str:
        q_emb_str = ','.join(map(str, q_emb))
        if where_str:
            where_str = f'PREWHERE {where_str}'
        else:
            where_str = ''
        q_str = f'\n            SELECT {self.config.column_map['text']}, \n                {self.config.column_map['metadata']}, dist\n            FROM {self.config.database}.{self.config.table}\n            {where_str}\n            ORDER BY distance({self.config.column_map['vector']}, [{q_emb_str}]) \n                AS dist {self.dist_order}\n            LIMIT {topk}\n            '
        return q_str

    def similarity_search(self, query: str, k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Document]:
        """Perform a similarity search with MyScale

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
        return self.similarity_search_by_vector(self._embeddings.embed_query(query), k, where_str, **kwargs)

    def similarity_search_by_vector(self, embedding: List[float], k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Document]:
        """Perform a similarity search with MyScale by vectors

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
        q_str = self._build_qstr(embedding, k, where_str)
        try:
            return [Document(page_content=r[self.config.column_map['text']], metadata=r[self.config.column_map['metadata']]) for r in self.client.query(q_str).named_results()]
        except Exception as e:
            logger.error(f'\x1b[91m\x1b[1m{type(e)}\x1b[0m \x1b[95m{str(e)}\x1b[0m')
            return []

    def similarity_search_with_relevance_scores(self, query: str, k: int=4, where_str: Optional[str]=None, **kwargs: Any) -> List[Tuple[Document, float]]:
        """Perform a similarity search with MyScale

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
            List[Document]: List of documents most similar to the query text
            and cosine distance in float for each.
            Lower score represents more similarity.
        """
        q_str = self._build_qstr(self._embeddings.embed_query(query), k, where_str)
        try:
            return [(Document(page_content=r[self.config.column_map['text']], metadata=r[self.config.column_map['metadata']]), r['dist']) for r in self.client.query(q_str).named_results()]
        except Exception as e:
            logger.error(f'\x1b[91m\x1b[1m{type(e)}\x1b[0m \x1b[95m{str(e)}\x1b[0m')
            return []

    def drop(self) -> None:
        """
        Helper function: Drop data
        """
        self.client.command(f'DROP TABLE IF EXISTS {self.config.database}.{self.config.table}')

    def delete(self, ids: Optional[List[str]]=None, where_str: Optional[str]=None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        assert not (ids is None and where_str is None), 'You need to specify where to be deleted! Either with `ids` or `where_str`'
        conds = []
        if ids and len(ids) > 0:
            id_list = ', '.join([f"'{id}'" for id in ids])
            conds.append(f'{self.config.column_map['id']} IN ({id_list})')
        if where_str:
            conds.append(where_str)
        assert len(conds) > 0
        where_str_final = ' AND '.join(conds)
        qstr = f'DELETE FROM {self.config.database}.{self.config.table} WHERE {where_str_final}'
        try:
            self.client.command(qstr)
            return True
        except Exception as e:
            logger.error(str(e))
            return False

    @property
    def metadata_column(self) -> str:
        return self.config.column_map['metadata']