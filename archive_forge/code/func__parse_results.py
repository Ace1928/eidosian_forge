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
def _parse_results(self, cmd, res, **kwargs):
    if get_protocol_version(self.client) in ['3', 3]:
        return res
    else:
        return self._RESP2_MODULE_CALLBACKS[cmd](res, **kwargs)