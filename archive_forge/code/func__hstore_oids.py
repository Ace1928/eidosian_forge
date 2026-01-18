from __future__ import annotations
import collections.abc as collections_abc
import logging
import re
from typing import cast
from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from ... import types as sqltypes
from ... import util
from ...util import FastIntFlag
from ...util import parse_user_argument_for_enum
@util.memoized_instancemethod
def _hstore_oids(self, dbapi_connection):
    extras = self._psycopg2_extras
    oids = extras.HstoreAdapter.get_oids(dbapi_connection)
    if oids is not None and oids[0]:
        return oids[0:2]
    else:
        return None