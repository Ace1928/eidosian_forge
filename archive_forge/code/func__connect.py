import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, List, Optional
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.util.annotations import PublicAPI
@contextmanager
def _connect(connection_factory: Callable[[], Connection]) -> Iterator[Cursor]:
    connection = connection_factory()
    _check_connection_is_dbapi2_compliant(connection)
    try:
        cursor = connection.cursor()
        _check_cursor_is_dbapi2_compliant(cursor)
        yield cursor
        connection.commit()
    except Exception:
        try:
            connection.rollback()
        except Exception as e:
            if isinstance(e, AttributeError) or e.__class__.__name__ == 'NotSupportedError':
                pass
        raise
    finally:
        connection.close()