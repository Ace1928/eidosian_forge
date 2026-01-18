from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def __get_db(self, config: KineticaSettings) -> Any:
    try:
        from gpudb import GPUdb
    except ImportError:
        raise ImportError('Could not import Kinetica python API. Please install it with `pip install gpudb==7.2.0.1`.')
    options = GPUdb.Options()
    options.username = config.username
    options.password = config.password
    options.skip_ssl_cert_verification = True
    return GPUdb(host=config.host, options=options)