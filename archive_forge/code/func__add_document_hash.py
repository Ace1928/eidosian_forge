import itertools
import time
from typing import Dict, List, Optional, Union
from redis.client import Pipeline
from redis.utils import deprecated_function
from ..helpers import get_protocol_version, parse_to_dict
from ._util import to_string
from .aggregation import AggregateRequest, AggregateResult, Cursor
from .document import Document
from .query import Query
from .result import Result
from .suggestion import SuggestionParser
def _add_document_hash(self, doc_id, conn=None, score=1.0, language=None, replace=False):
    """
        Internal add_document_hash used for both batch and single doc indexing
        """
    args = [ADDHASH_CMD, self.index_name, doc_id, score]
    if replace:
        args.append('REPLACE')
    if language:
        args += ['LANGUAGE', language]
    if conn is not None:
        return conn.execute_command(*args)
    return self.execute_command(*args)