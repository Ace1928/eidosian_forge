from typing import Any, Iterator, List, Optional, Sequence, Tuple, cast
from langchain_core._api.deprecation import deprecated
from langchain_core.stores import BaseStore, ByteStore
@deprecated('0.0.1', alternative='UpstashRedisByteStore')
class UpstashRedisStore(_UpstashRedisStore):
    """
    BaseStore implementation using Upstash Redis
    as the underlying store to store strings.

    Deprecated in favor of the more generic UpstashRedisByteStore.
    """