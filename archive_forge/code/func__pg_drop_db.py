import time
from ... import exc
from ... import inspect
from ... import text
from ...testing import warn_test_suite
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_post_tables
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import prepare_for_drop_tables
from ...testing.provision import set_default_schema_on_connection
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@drop_db.for_db('postgresql')
def _pg_drop_db(cfg, eng, ident):
    with eng.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
        with conn.begin():
            conn.execute(text('select pg_terminate_backend(pid) from pg_stat_activity where usename=current_user and pid != pg_backend_pid() and datname=:dname'), dict(dname=ident))
            conn.exec_driver_sql('DROP DATABASE %s' % ident)