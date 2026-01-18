import os
import re
from ... import exc
from ...engine import url as sa_url
from ...testing.provision import create_db
from ...testing.provision import drop_db
from ...testing.provision import follower_url_from_main
from ...testing.provision import generate_driver_url
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import run_reap_dbs
from ...testing.provision import stop_test_class_outside_fixtures
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@run_reap_dbs.for_db('sqlite')
def _reap_sqlite_dbs(url, idents):
    log.info('db reaper connecting to %r', url)
    log.info('identifiers in file: %s', ', '.join(idents))
    url = sa_url.make_url(url)
    for ident in idents:
        for drivername in _drivernames:
            _drop_dbs_w_ident(url.database, drivername, ident)