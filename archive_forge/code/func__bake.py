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
def _bake(self, session):
    query = self._as_query(session)
    query.session = None
    statement = query._statement_20()
    if statement._compile_options._bake_ok:
        self._bakery[self._effective_key(session)] = (query, statement)
    return (query, statement)