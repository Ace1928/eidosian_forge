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
def _as_query(self):
    q = self.bq._as_query(self.session).params(self._params)
    for fn in self._post_criteria:
        q = fn(q)
    return q