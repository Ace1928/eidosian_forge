import collections.abc as collections_abc
import logging
from .. import exc as sa_exc
from .. import util
from ..orm import exc as orm_exc
from ..orm.query import Query
from ..orm.session import Session
from ..sql import func
from ..sql import literal_column
from ..sql import util as sql_util
def _with_lazyload_options(self, options, effective_path, cache_path=None):
    """Cloning version of _add_lazyload_options."""
    q = self._clone()
    q._add_lazyload_options(options, effective_path, cache_path=cache_path)
    return q