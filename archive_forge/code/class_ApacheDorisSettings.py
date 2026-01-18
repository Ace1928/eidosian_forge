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
class ApacheDorisSettings(BaseSettings):
    """Apache Doris client configuration.

    Attributes:
        apache_doris_host (str) : An URL to connect to frontend.
                             Defaults to 'localhost'.
        apache_doris_port (int) : URL port to connect with HTTP. Defaults to 9030.
        username (str) : Username to login. Defaults to 'root'.
        password (str) : Password to login. Defaults to None.
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on.
                      Defaults to 'langchain'.

        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'embedding': 'text_embedding',
                                    'document': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.
    """
    host: str = 'localhost'
    port: int = 9030
    username: str = 'root'
    password: str = ''
    column_map: Dict[str, str] = {'id': 'id', 'document': 'document', 'embedding': 'embedding', 'metadata': 'metadata'}
    database: str = 'default'
    table: str = 'langchain'

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    class Config:
        env_file = '.env'
        env_prefix = 'apache_doris_'
        env_file_encoding = 'utf-8'