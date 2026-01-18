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
@post_configure_engine.for_db('sqlite')
def _sqlite_post_configure_engine(url, engine, follower_ident):
    from sqlalchemy import event
    if follower_ident:
        attach_path = f'{follower_ident}_{engine.driver}_test_schema.db'
    else:
        attach_path = f'{engine.driver}_test_schema.db'

    @event.listens_for(engine, 'connect')
    def connect(dbapi_connection, connection_record):
        dbapi_connection.execute(f'ATTACH DATABASE "{attach_path}" AS test_schema')

    @event.listens_for(engine, 'engine_disposed')
    def dispose(engine):
        """most databases should be dropped using
        stop_test_class_outside_fixtures

        however a few tests like AttachedDBTest might not get triggered on
        that main hook

        """
        if os.path.exists(attach_path):
            os.remove(attach_path)
        filename = engine.url.database
        if filename and filename != ':memory:' and os.path.exists(filename):
            os.remove(filename)