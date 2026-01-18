from __future__ import annotations
import collections
import logging
from . import config
from . import engines
from . import util
from .. import exc
from .. import inspect
from ..engine import url as sa_url
from ..sql import ddl
from ..sql import schema
def _configs_for_db_operation():
    hosts = set()
    for cfg in config.Config.all_configs():
        cfg.db.dispose()
    for cfg in config.Config.all_configs():
        url = cfg.db.url
        backend = url.get_backend_name()
        host_conf = (backend, url.username, url.host, url.database)
        if host_conf not in hosts:
            yield cfg
            hosts.add(host_conf)
    for cfg in config.Config.all_configs():
        cfg.db.dispose()