import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
def _setup_for_connection(self, sql_connection, engine_kwargs, maker_kwargs):
    if sql_connection is None:
        raise exception.CantStartEngineError('No sql_connection parameter is established')
    engine = engines.create_engine(sql_connection=sql_connection, **engine_kwargs)
    for hook in self._facade_cfg['on_engine_create']:
        hook(engine)
    sessionmaker = orm.get_maker(engine=engine, **maker_kwargs)
    return (engine, sessionmaker)