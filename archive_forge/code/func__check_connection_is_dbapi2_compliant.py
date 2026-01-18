import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
def _check_connection_is_dbapi2_compliant(connection) -> None:
    for attr in ('close', 'commit', 'cursor'):
        if not hasattr(connection, attr):
            raise ValueError(f'Your `connection_factory` created a `Connection` object without a {attr!r} method, but this method is required by the Python DB API2 specification. Check that your database connector is DB API2-compliant. To learn more, read https://peps.python.org/pep-0249/.')