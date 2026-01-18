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
def _retrieve_baked_query(self, session):
    query = self._bakery.get(self._effective_key(session), None)
    if query is None:
        query = self._as_query(session)
        self._bakery[self._effective_key(session)] = query.with_session(None)
    return query.with_session(session)