import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
def _query_to_dict(query: str) -> Dict[str, str]:
    return {pair[0]: pair[1] for pair in (pair.split('=') for pair in filter(None, query.split('&')))}