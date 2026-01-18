from ... import create_engine
from ... import exc
from ... import inspect
from ...engine import url as sa_url
from ...testing.provision import configure_follower
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_post_tables
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import follower_url_from_main
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import run_reap_dbs
from ...testing.provision import set_default_schema_on_connection
from ...testing.provision import stop_test_class_outside_fixtures
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import update_db_opts
def _ora_drop_ignore(conn, dbname):
    try:
        conn.exec_driver_sql('drop user %s cascade' % dbname)
        log.info('Reaped db: %s', dbname)
        return True
    except exc.DatabaseError as err:
        log.warning("couldn't drop db: %s", err)
        return False