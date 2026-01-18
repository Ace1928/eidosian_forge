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
@set_default_schema_on_connection.for_db('postgresql')
def _postgresql_set_default_schema_on_connection(cfg, dbapi_connection, schema_name):
    existing_autocommit = dbapi_connection.autocommit
    dbapi_connection.autocommit = True
    cursor = dbapi_connection.cursor()
    cursor.execute("SET SESSION search_path='%s'" % schema_name)
    cursor.close()
    dbapi_connection.autocommit = existing_autocommit